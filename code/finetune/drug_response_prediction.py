import copy
import json
import logging
import os
import random
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import transformers
from peft import LoraConfig, get_peft_model
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Hotfix for transformers
if not hasattr(transformers.utils, "LossKwargs"):
    class LossKwargs: pass
    transformers.utils.LossKwargs = LossKwargs

# 添加路径以导入自定义模块
sys.path.append("../")
from omix.model.omni_fusion import (
    OmniFusionBlock, TrainableTextEncoder)
from omix.model.model_multi import PerformerModel
from omix.preprocess_bulk import Preprocessor as Preprocessor_RNA
from omix.preprocess_rppa import Preprocessor as Preprocessor_Protein
from omix.tokenizer import GeneVocab, tokenize_and_pad_batch

# 忽略部分警告
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ====================================================================================
# 1. 配置参数
# ====================================================================================
OPTION = 'RPMT' # 必须包含所有需要的模态
MODALITIES_ORDER = ['RNA', 'Protein', 'METHYL', 'TEXT']
MODALITY_CHARS = {'RNA': 'R', 'Protein': 'P', 'METHYL': 'M', 'TEXT': 'T'}

# ['RNA', 'Protein', 'METHYL'], ['RNA', 'Protein', 'TEXT'], ['RNA', 'METHYL', 'TEXT'], ['Protein', 'METHYL', 'TEXT']
MODALITIES_TO_KEEP = ['RNA', 'Protein', 'METHYL', 'TEXT']

freeze_text = True
freeze_omics = True
freeze_fusion = False
use_gate_loss = False
# Drug Name: e.g., 'LAPATINIB', 'ERLOTINIB', 'PACLITAXEL', 'NILOTINIB', 'SORAFENIB', 'IRINOTECAN', 'TOPOTECAN'
DRUG_NAME = "ERLOTINIB"
print('using modality:', OPTION)
print('freeze_text:', freeze_text)
print('freeze_omics:', freeze_omics)
print('freeze_fusion:', freeze_fusion)
print('use_gate_loss:', use_gate_loss)
print('modality to keep:', MODALITIES_TO_KEEP)

# 路径设置
LOAD_MODEL_PATH = "../pretrain/save/omix_t_pretrain"
model_epoch = "model_e5.pt"
DATA_ROOT_DIR = "../../data/CCLE2019/preprocessed"
YOUTU_PATH = "omix/Youtu_embedding" 
TEXT_EMBEDDING_FILE = '../../data/pretraining_data/generated_files/textual_annotations_ccle2019.json'

gpu_index = 7
DEVICE = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(gpu_index)

# Hyperparameters
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
EPOCHS = 100
N_SPLITS = 5 
DROPOUT = 0.2
MAX_SEQ_LEN_PROTEIN = 17007 + 1
MAX_SEQ_LEN_RNA = 19205 + 1
MAX_SEQ_LEN_METHYL = 12920 + 1
MAX_SEQ_LEN_TEXT = 1024 
PAD_TOKEN = "<pad>"
PAD_VALUE = 51

print('model dir:', LOAD_MODEL_PATH, model_epoch)
print(f"Running Drug Response Finetuning for {DRUG_NAME} with OPTION: {OPTION}")
print(f"Dropout {DROPOUT} | BATCH_SIZE {BATCH_SIZE} | LEARNING_RATE {LEARNING_RATE} | GRADIENT_ACCUMULATION_STEPS {GRADIENT_ACCUMULATION_STEPS}")

# ====================================================================================
# 2. 辅助函数：组合计算与数据对齐
# ====================================================================================

def get_modality_combinations(modality_str):
    chars = list(modality_str)
    all_combs = []
    for r in range(1, len(chars) + 1):
        all_combs.extend(combinations(chars, r))
    sorted_combs = [''.join(sorted(comb)) for comb in all_combs]
    full_modality_str = ''.join(sorted(chars))
    # 将全模态移到第一位 (idx 0)
    if full_modality_str in sorted_combs:
        sorted_combs.remove(full_modality_str)
        sorted_combs.insert(0, full_modality_str)
    return {comb: i for i, comb in enumerate(sorted_combs)}, sorted_combs

# 计算组合映射
ALL_CHARS = "".join(sorted([MODALITY_CHARS[m] for m in MODALITIES_ORDER]))
COMBINATION_MAP, COMBINATION_LIST = get_modality_combinations(ALL_CHARS)
NUM_COMBINATIONS = len(COMBINATION_MAP)


print(f"Total Modality Combinations: {NUM_COMBINATIONS}")
print(f"Combinations: {COMBINATION_LIST}")

def create_flexmoe_input_for_cv(all_sample_ids, metadata_df, modalities_data, modalities_order, modalities_to_keep, tokenizer=None, max_len_text=1024):
    """
    灵活处理任意数量的模态数据，进行对齐、填充，并为懒加载模态创建映射。
    """
    logger = logging.getLogger()
    print("Aligning all modalities and creating dataset inputs...")

    all_sample_ids_lower = [sid.lower() for sid in all_sample_ids]
    id_to_idx = {sid: i for i, sid in enumerate(all_sample_ids_lower)}
    n_patients_initial = len(all_sample_ids)

    omics_data_dict = {}
    modality_to_order_idx = {name: i for i, name in enumerate(modalities_order)}
    observed_idx_arr = np.zeros((n_patients_initial, len(modalities_order)), dtype=bool)
    
    # 1. 填充数据
    for mod_name, mod_data in modalities_data.items():
        if mod_name not in modalities_order: continue
        
        mod_order_idx = modality_to_order_idx[mod_name]
        data_type = mod_data.get('type', 'omics')
        print(f"  - Processing '{mod_name}' (type: {data_type})...")

        if data_type == 'text_raw':
            ptids_in_modality = [str(pid).lower() for pid in mod_data['df'].index.tolist()]
            source_lookup = {str(pid).lower(): i for i, pid in enumerate(mod_data['df'].index)}
        else:
            ptids_in_modality = [str(pid).lower() for pid in mod_data['adata'].obs.index.tolist()]
            source_lookup = {str(pid).lower(): i for i, pid in enumerate(mod_data['adata'].obs.index)}

        global_indices = [id_to_idx[ptid] for ptid in ptids_in_modality if ptid in id_to_idx]
        if not global_indices: continue
        
        observed_idx_arr[global_indices, mod_order_idx] = True
        valid_ptids_lower = [all_sample_ids_lower[i] for i in global_indices]
        source_indices = [source_lookup[pid] for pid in valid_ptids_lower]

        if data_type == 'omics':
            tokenized_data = mod_data['tokenized']
            vocab = mod_data['vocab']
            
            genes_full = np.full((n_patients_initial, tokenized_data['genes'].shape[1]), vocab[PAD_TOKEN], dtype=np.int64)
            values_full = np.full((n_patients_initial, tokenized_data['values'].shape[1]), PAD_VALUE, dtype=np.float32)
            
            genes_full[global_indices] = tokenized_data['genes'][source_indices]
            values_full[global_indices] = tokenized_data['values'][source_indices]
            
            omics_data_dict[f'{mod_name}_genes'] = genes_full
            omics_data_dict[f'{mod_name}_values'] = values_full
        
        elif data_type == 'text_raw':
            if tokenizer is None: raise ValueError(f"Tokenizer required for {mod_name}")
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            input_ids_full = np.full((n_patients_initial, max_len_text), pad_id, dtype=np.int64)
            attn_mask_full = np.zeros((n_patients_initial, max_len_text), dtype=np.int64)
            
            raw_texts = mod_data['df'].iloc[source_indices]['text'].astype(str).tolist()
            encoded = tokenizer(raw_texts, padding='max_length', truncation=True, max_length=max_len_text, return_tensors='np')
            
            input_ids_full[global_indices] = encoded['input_ids']
            attn_mask_full[global_indices] = encoded['attention_mask']
            
            omics_data_dict[f'{mod_name}_input_ids'] = input_ids_full
            omics_data_dict[f'{mod_name}_attention_masks'] = attn_mask_full

    # 2. 样本过滤 (Filter logic)
    keep_indices = [i for i, m in enumerate(modalities_order) if m in modalities_to_keep]

    # 强制将不需要模态的 observed 设为 False (防止脏数据干扰)
    ignore_indices = [i for i, m in enumerate(modalities_order) if m not in modalities_to_keep]
    if ignore_indices:
        observed_idx_arr[:, ignore_indices] = False

    valid_observed_counts = observed_idx_arr[:, keep_indices].sum(axis=1)
    valid_sample_indices = np.where(valid_observed_counts > 0)[0]
    
    # === [改进点] 打印过滤详情 ===
    if len(valid_sample_indices) < n_patients_initial:
        n_dropped = n_patients_initial - len(valid_sample_indices)
        print(f"    [FILTERING] Dropping {n_dropped} samples (No valid data in {modalities_to_keep}).")
        print(f"    [FILTERING] Retaining {len(valid_sample_indices)} / {n_patients_initial} samples.")
        
        for key in list(omics_data_dict.keys()):
            omics_data_dict[key] = omics_data_dict[key][valid_sample_indices]
        observed_idx_arr = observed_idx_arr[valid_sample_indices]
        final_sample_ids = [all_sample_ids[i] for i in valid_sample_indices]
    else:
        print("    [FILTERING] No samples dropped.")
        final_sample_ids = all_sample_ids
        
    n_patients = len(final_sample_ids)

    # 3. 生成 Modality Combination Index
    mod_name_to_char = {name: MODALITY_CHARS[name] for name in modalities_data.keys()}
    modality_comb_arr = np.zeros(n_patients, dtype=np.int64)
    
    for i in range(n_patients):
        valid_chars = []
        for m_idx in keep_indices:
            if observed_idx_arr[i, m_idx]:
                mod_name = modalities_order[m_idx]
                valid_chars.append(mod_name_to_char[mod_name])
        comb_str = "".join(sorted(valid_chars))
        modality_comb_arr[i] = COMBINATION_MAP.get(comb_str, 0) # Default to 0 (Full) if weirdness, usually safe

    # 4. 整理 Labels
    temp_labels = metadata_df['label']
    temp_labels.index = temp_labels.index.str.lower()
    final_sample_ids_lower = [sid.lower() for sid in final_sample_ids]
    aligned_labels = temp_labels.loc[final_sample_ids_lower].values.astype(np.float32)

    metadata_dict = {
        'observed_idx': observed_idx_arr,
        'modality_comb': modality_comb_arr,
        'labels': aligned_labels
    }
    return omics_data_dict, metadata_dict, final_sample_ids

# ====================================================================================
# 3. 数据加载与 Dataset
# ====================================================================================

def tokenize_omics(df, vocab, max_len, omics_type):
    """Aux helper to tokenize omics dataframe"""
    adata = anndata.AnnData(df)
    adata.var['gene_name'] = adata.var.index

    # 假设这里预处理逻辑一致，统称为 Preprocessor_RNA 以简化（根据你原代码）
    # 如果Protein特别需要 RPPA Preprocessor，请在此区分
    if omics_type != 'Protein':
        pp = Preprocessor_RNA(use_key="X", filter_gene_by_counts=False, filter_cell_by_counts=False,
                            normalize_total=False, log1p=False, binning=51, result_binned_key="X_binned")
    else:
        pp = Preprocessor_Protein(use_key="X", filter_gene_by_counts=False, filter_cell_by_counts=False,
                                  normalize_total=False, log1p=False, binning=51, result_binned_key="X_binned")
    pp(adata)
    
    gene_ids = np.array(vocab(adata.var_names.tolist()), dtype=int)
    tokenized = tokenize_and_pad_batch(
        adata.layers["X_binned"], gene_ids, max_len=max_len, 
        vocab=vocab, pad_token=PAD_TOKEN, pad_value=PAD_VALUE, append_cls=True, include_zero_gene=False
    )
    return tokenized, adata

def load_data_union(data_dir, drug_name, text_json_path, vocab_dict):

    """加载所有数据，不取交集，而是准备并集数据字典"""
    print(f"Loading labels for {drug_name}...")
    labels_df = pd.read_csv(os.path.join(data_dir, 'drug_resposne_labels.csv'), index_col=0)
    labels_df = labels_df[labels_df['DRUG_NAME'] == drug_name]
    if 'SAMPLE_NAME' in labels_df.columns:
        labels_df.set_index('SAMPLE_NAME', inplace=True)
    labels_df.index = labels_df.index.str.split('_').str[0].str.upper()
    labels_df = labels_df[~labels_df.index.duplicated(keep='first')]

    modalities_data = {}
    
    # RNA
    print("Loading RNA...")
    df_rna = pd.read_csv(os.path.join(data_dir, 'rna.csv'), index_col=0)
    df_rna.index = df_rna.index.str.split('_').str[0].str.upper()
    df_rna = df_rna[~df_rna.index.duplicated(keep='first')]

    tok_rna, adata_rna = tokenize_omics(df_rna, vocab_dict['RNA'], MAX_SEQ_LEN_RNA, omics_type='RNA')
    modalities_data['RNA'] = {
        'type': 'omics', 'adata': adata_rna, 'tokenized': tok_rna, 'vocab': vocab_dict['RNA'], 'char': 'R'
    }

    # Protein
    print("Loading Protein...")
    df_prot = pd.read_csv(os.path.join(data_dir, 'rppa.csv'), index_col=0)
    df_prot.index = df_prot.index.str.split('_').str[0].str.upper()
    df_prot = df_prot[~df_prot.index.duplicated(keep='first')]
    tok_prot, adata_prot = tokenize_omics(df_prot, vocab_dict['Protein'], MAX_SEQ_LEN_PROTEIN, omics_type='Protein')
    modalities_data['Protein'] = {
        'type': 'omics', 'adata': adata_prot, 'tokenized': tok_prot, 'vocab': vocab_dict['Protein'], 'char': 'P'
    }

    # Methyl
    print("Loading Methylation...")
    df_meth = pd.read_csv(os.path.join(data_dir, 'methylation.csv'), index_col=0)
    df_meth.index = df_meth.index.str.split('_').str[0].str.upper()
    df_meth = df_meth[~df_meth.index.duplicated(keep='first')]

    tok_meth, adata_meth = tokenize_omics(df_meth, vocab_dict['METHYL'], MAX_SEQ_LEN_METHYL, omics_type='METHYL')
    modalities_data['METHYL'] = {
        'type': 'omics', 'adata': adata_meth, 'tokenized': tok_meth, 'vocab': vocab_dict['METHYL'], 'char': 'M'
    }

    # Text
    print("Loading Text...")
    with open(text_json_path, 'r') as f:
        raw_text_dict = json.load(f)
    cleaned_text = {}
    for k, v in raw_text_dict.items():
        simple_id = k.split('-')[0].upper()
        cleaned_text[simple_id] = v
    df_text = pd.DataFrame.from_dict(cleaned_text, orient='index', columns=['text'])
    modalities_data['TEXT'] = {
        'type': 'text_raw', 'df': df_text, 'char': 'T'
    }

    # Initial Union of Sample IDs (Intersection of Label and (Union of Mods))
    all_mod_ids = set()
    for m in modalities_data:
        if m == 'TEXT': all_mod_ids.update(df_text.index)
        else: all_mod_ids.update(modalities_data[m]['adata'].obs.index)
    
    # 我们只关心有 Label 的样本
    common_ids = sorted(list(set(labels_df.index).intersection(all_mod_ids)))
    print(f"Total samples with Labels and at least one modality: {len(common_ids)}")

    # Construct metadata DF for splitting logic
    metadata_df = pd.DataFrame(index=common_ids)
    metadata_df['label'] = labels_df.loc[common_ids, 'aac_published']
    
    # Determine 'type' (full vs partial) for splitting
    # 这里需要临时构建个observed matrix来判断
    is_full = np.ones(len(common_ids), dtype=bool)
    modality_strings = []
    
    for sid in common_ids:
        mods_present = []
        for m in MODALITIES_ORDER:
            # Check existence
            if m == 'TEXT': exists = sid in df_text.index
            else: exists = sid in modalities_data[m]['adata'].obs.index
            
            if not exists: is_full[common_ids.index(sid)] = False
            if exists: mods_present.append(MODALITY_CHARS[m])
        modality_strings.append(",".join([MODALITIES_ORDER[i] for i, x in enumerate(mods_present) if x]))

    metadata_df['type'] = ['full' if x else 'partial' for x in is_full]
    metadata_df['modalities'] = modality_strings

    return modalities_data, metadata_df, common_ids

class OmniFusionBlockDataset(Dataset):
    def __init__(self, omics_data, metadata, vocab_dict):
        self.omics_data = omics_data
        self.observed_idx = metadata['observed_idx']
        self.modality_comb = metadata['modality_comb']
        self.labels = metadata['labels']
        self.length = len(self.labels)
        self.vocab_dict = vocab_dict

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = {}
        # RNA
        if 'RNA_genes' in self.omics_data:
            item['RNA'] = (
                torch.tensor(self.omics_data['RNA_genes'][idx]),
                torch.tensor(self.omics_data['RNA_values'][idx]),
                torch.tensor(self.omics_data['RNA_genes'][idx]).eq(self.vocab_dict['RNA'][PAD_TOKEN]) # Mask (Assuming 0 is PAD_ID in this context check logic)
            )
        # Protein
        if 'Protein_genes' in self.omics_data:
            item['Protein'] = (
                torch.tensor(self.omics_data['Protein_genes'][idx]),
                torch.tensor(self.omics_data['Protein_values'][idx]),
                torch.tensor(self.omics_data['Protein_genes'][idx]).eq(self.vocab_dict['Protein'][PAD_TOKEN])
            )
        # Methyl
        if 'METHYL_genes' in self.omics_data:
            item['METHYL'] = (
                torch.tensor(self.omics_data['METHYL_genes'][idx]),
                torch.tensor(self.omics_data['METHYL_values'][idx]),
                torch.tensor(self.omics_data['METHYL_genes'][idx]).eq(self.vocab_dict['METHYL'][PAD_TOKEN])
            )
        # Text
        if 'TEXT_input_ids' in self.omics_data:
            item['TEXT'] = (
                torch.tensor(self.omics_data['TEXT_input_ids'][idx]),
                torch.tensor(self.omics_data['TEXT_attention_masks'][idx])
            )
            
        item['observed'] = torch.tensor(self.observed_idx[idx], dtype=torch.bool)
        item['comb_idx'] = torch.tensor(self.modality_comb[idx], dtype=torch.long)
        item['label'] = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return item

# ====================================================================================
# 4. CV Splitting Logic (Refactored)
# ====================================================================================

def get_stratified_kfold_splits(metadata_df, n_splits, random_state):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    sample_ids = metadata_df.index
    stratify_labels = metadata_df['type'] # Stratify by 'full' vs 'partial'
    
    splits = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(sample_ids, stratify_labels)):
        splits.append({
            'fold': fold, 
            'train_ids': sample_ids[train_idx].tolist(), 
            'test_ids': sample_ids[test_idx].tolist()
        })
    return splits

def create_cv_splits_after_preprocessing(metadata_df, n_splits=5, random_state=42, filter_val_for_common=True):
    """
    在预处理后创建交叉验证划分，并保存每个样本的模态信息
    """
    CV_SPLIT_FILE = Path(DATA_ROOT_DIR) / f"cv_splits_drug_{DRUG_NAME}_{n_splits}fold.csv"
    
    if CV_SPLIT_FILE.exists():
        print(f"Loading existing CV splits from {CV_SPLIT_FILE}")
        split_df = pd.read_csv(CV_SPLIT_FILE)
        cv_splits = []
        for fold in range(n_splits):
            train_ids = split_df[(split_df['fold'] == fold) & (split_df['split_type'] == 'train')]['sample_id'].tolist()
            test_ids = split_df[(split_df['fold'] == fold) & (split_df['split_type'] == 'test')]['sample_id'].tolist()
            cv_splits.append({'fold': fold, 'train_ids': train_ids, 'test_ids': test_ids})
    else:
        print(f"Creating new CV splits and saving to {CV_SPLIT_FILE}")
        exit()
        # cv_splits = get_stratified_kfold_splits(metadata_df, n_splits, random_state)
        
        # records = []
        # for fold_info in cv_splits:
        #     fold = fold_info['fold']
        #     for sample_id in fold_info['train_ids']:
        #         modalities = metadata_df.loc[sample_id, 'modalities'] if sample_id in metadata_df.index else ''
        #         sample_type = metadata_df.loc[sample_id, 'type'] if sample_id in metadata_df.index else ''
        #         records.append({
        #             'fold': fold, 
        #             'split_type': 'train', 
        #             'sample_id': sample_id,
        #             'modalities': modalities,
        #             'type': sample_type
        #         })
        #     for sample_id in fold_info['test_ids']:
        #         modalities = metadata_df.loc[sample_id, 'modalities'] if sample_id in metadata_df.index else ''
        #         sample_type = metadata_df.loc[sample_id, 'type'] if sample_id in metadata_df.index else ''
        #         records.append({
        #             'fold': fold, 
        #             'split_type': 'test', 
        #             'sample_id': sample_id,
        #             'modalities': modalities,
        #             'type': sample_type
        #         })
        # pd.DataFrame(records).to_csv(CV_SPLIT_FILE, index=False)
    
    if filter_val_for_common:
        print("Filtering validation/test sets to keep only full modality samples.")
        
        # 1. 找到所有'full'模态的样本ID
        full_modality_sample_ids = set(metadata_df[metadata_df['type'] == 'full'].index)
        
        # 2. 遍历每个fold，并筛选其 test_ids
        filtered_cv_splits = []
        for fold_info in cv_splits:
            original_test_ids = set(fold_info['test_ids'])
            
            # 筛选：取原始test_ids和full_modality_sample_ids的交集
            filtered_test_ids = list(original_test_ids.intersection(full_modality_sample_ids))
            
            print(f"  Fold {fold_info['fold']}: Validation set size changed from {len(original_test_ids)} to {len(filtered_test_ids)}")

            # 创建一个新的split_info字典，保留原始的train_ids，但使用筛选后的test_ids
            filtered_fold_info = {
                'fold': fold_info['fold'],
                'train_ids': fold_info['train_ids'], # 训练集保持不变，包含不完整样本
                'test_ids': filtered_test_ids
            }
            filtered_cv_splits.append(filtered_fold_info)
        
        # 3. 返回筛选后的划分
        return filtered_cv_splits
    else:
        # 如果不进行筛选，则返回原始的划分
        print("Using original validation/test sets with both full and partial samples.")
        return cv_splits

# ====================================================================================
# 5. 模型定义 (Modified for Missing Embeddings)
# ====================================================================================

class DrugResponsePredictor(nn.Module):
    def __init__(self, encoder_dict, fusion_model, embed_dim, vocab_mod_map, dropout, num_combinations):
        super().__init__()
        # self.encoder_dict = encoder_dict
        self.encoder_dict = nn.ModuleDict(encoder_dict)
        self.fusion_model = fusion_model
        self.vocab_mod_map = vocab_mod_map
        self.num_modalities = 4
        self.embed_dim = embed_dim
        self.num_combinations = num_combinations
        
        self.missing_embeds_dict = nn.ParameterDict()
        for mod in MODALITIES_ORDER:
            self.missing_embeds_dict[mod] = nn.Parameter(
                torch.randn(num_combinations, 1, embed_dim)
            )

        head_in_dim = embed_dim * self.num_modalities
        self.head = nn.Sequential(
            nn.Linear(head_in_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )

    def load_missing_embeds(self, checkpoint, device):
        if 'missing_embeds' not in checkpoint:
            print("[Warning] No 'missing_embeds' found in checkpoint. Using random init.")
            return

        print("Loading and slicing missing_embeds from pretrained model...")
        pretrained_missing = checkpoint['missing_embeds']
        
        for mod_name, param in self.missing_embeds_dict.items():
            if mod_name in pretrained_missing:
                full_tensor = pretrained_missing[mod_name].to(device)
                mod_idx = self.vocab_mod_map.get(mod_name)
                if mod_idx is None: continue
                
                sliced_tensor = full_tensor[:, mod_idx, :, :] 
                if sliced_tensor.dim() == 2: 
                    sliced_tensor = sliced_tensor.unsqueeze(1)
                    
                if sliced_tensor.shape == param.shape:
                    param.data.copy_(sliced_tensor)
                    print(f"  Loaded {mod_name} missing embedding. Shape: {sliced_tensor.shape}")
                else:
                    print(f"  [Shape Mismatch] {mod_name}: Checkpoint {sliced_tensor.shape} != Model {param.shape}")

    def run_encoder(self, mod_name, batch_data, device):
        """Running Encoder to get CLS (Pre-Adapter)"""
        # 注意：这里的 batch_data 已经在 forward 里移动到 device 了，但再次 .to(device) 是安全的
        if mod_name == 'TEXT':
            input_ids, attn_mask = batch_data
            input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
            enc_out, _ = self.encoder_dict['TEXT'](input_ids, attn_mask)
            feat = enc_out[:, 0:1, :] if enc_out.dim() == 3 else enc_out.unsqueeze(1)
        else:
            genes, vals, mask = batch_data
            genes, vals, mask = genes.to(device), vals.to(device), mask.to(device)
            enc_out, _ = self.encoder_dict[mod_name](
                src=genes, values=vals, src_key_padding_mask=mask, return_seq_embedding=True
            )
            feat = enc_out[:, 0:1, :]
        return feat

    def run_adapter_safe(self, mod_name, feat):
        if hasattr(self.fusion_model, "module"):
            return self.fusion_model.module.run_adapter(mod_name, feat)
        return self.fusion_model.run_adapter(mod_name, feat)

    def forward(self, batch_dict):
        device = next(self.parameters()).device
        
        # 1. Mask 和 Comb Index 移到 GPU
        observed_mask = batch_dict['observed'].to(device) # [B, 4]
        comb_idx = batch_dict['comb_idx'].to(device)      # [B]
        # print('zzz')
        # print(comb_idx)
        # print(observed_mask)
        batch_size = observed_mask.shape[0]

        embeddings_list = []
        mod_types_list = []
        
        for i, mod in enumerate(MODALITIES_ORDER):
            # 初始化特征容器
            feat_tensor = torch.zeros(batch_size, 1, self.embed_dim, device=device)
            
            # present_indices 此时在 GPU 上
            present_indices = torch.where(observed_mask[:, i])[0]
            missing_indices = torch.where(~observed_mask[:, i])[0]
            
            # === 处理存在的样本 ===
            if len(present_indices) > 0:
                # 获取原始数据 (此时还在 CPU 上的 Tuple)
                sub_data_cpu = batch_dict[mod]
                
                # 【关键修复】：切片前，先把需要切片的数据移到 GPU
                # 这样 present_indices (GPU) 就可以切片 sub_data_gpu (GPU) 了
                sub_data_gpu = [t.to(device) for t in sub_data_cpu]
                
                if mod == 'TEXT':
                    sub_inputs = (
                        sub_data_gpu[0][present_indices], 
                        sub_data_gpu[1][present_indices]
                    )
                else:
                    sub_inputs = (
                        sub_data_gpu[0][present_indices], 
                        sub_data_gpu[1][present_indices], 
                        sub_data_gpu[2][present_indices]
                    )
                
                # 运行 Encoder
                raw_feat = self.run_encoder(mod, sub_inputs, device)
                feat_tensor[present_indices] = raw_feat

            # === 处理缺失的样本 ===
            if len(missing_indices) > 0:
                # print('fff')
                # print('missing!!!\n\n')
                missing_combs = comb_idx[missing_indices]
                # 参数 self.missing_embeds_dict 已经在 GPU 上，所以直接索引没问题
                miss_feat = self.missing_embeds_dict[mod][missing_combs] 
                feat_tensor[missing_indices] = miss_feat
            
            # === 统一过 Adapter ===
            feat_tensor = self.run_adapter_safe(mod, feat_tensor)
            
            embeddings_list.append(feat_tensor)
            
            # 构建 mod_types
            mod_id = self.vocab_mod_map[mod]
            mod_types_list.append(torch.full((batch_size, 1), mod_id, dtype=torch.long, device=device))

        # Fusion
        mod_types = torch.cat(mod_types_list, dim=1)
        expert_indices = comb_idx
        
        final_repr = self.fusion_model(
            *embeddings_list,
            expert_indices=expert_indices,
            mod_types=mod_types,
            return_cell_embedding=True
        )
        final_repr = final_repr.reshape(batch_size, -1)
        
        # Prediction
        pred = self.head(final_repr).squeeze(-1)
        gate_loss = self.fusion_model.gate_loss()
        return pred, gate_loss

# ====================================================================================
# 6. Training Functions
# ====================================================================================

def train_one_epoch(model, dataloader, optimizer, criterion, device, modalities_to_keep):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    for i, batch in enumerate(tqdm(dataloader, desc="Training")):
        targets = batch['label'].to(device)

        if modalities_to_keep is not None:
            # MODALITIES_ORDER 是全局变量 ['RNA', 'Protein', 'METHYL', 'TEXT']
            keep_mask = torch.tensor(
                [mod in modalities_to_keep for mod in MODALITIES_ORDER], 
                dtype=torch.bool, 
                device=device
            )
            # batch['observed'] 已经在 device 上或者即将被送到 device
            batch['observed'] = batch['observed'].to(device) & keep_mask

        preds, gate_loss = model(batch)
        loss = criterion(preds, targets)
        if use_gate_loss:
            loss = loss + 0.01 * gate_loss
        else:
            loss = loss + 0.00 * gate_loss
        
        loss = loss / GRADIENT_ACCUMULATION_STEPS
        loss.backward()
        
        if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device, modalities_to_keep):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            targets = batch['label'].to(device)

            if modalities_to_keep is not None:
                # MODALITIES_ORDER 是全局变量 ['RNA', 'Protein', 'METHYL', 'TEXT']
                keep_mask = torch.tensor(
                    [mod in modalities_to_keep for mod in MODALITIES_ORDER], 
                    dtype=torch.bool, 
                    device=device
                )
                # batch['observed'] 已经在 device 上或者即将被送到 device
                batch['observed'] = batch['observed'].to(device) & keep_mask

            preds, gate_loss= model(batch)
            loss = criterion(preds, targets)
            if use_gate_loss:
                loss = loss + 0.01 * gate_loss
            else:
                loss = loss + 0.00 * gate_loss
            
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    mse = mean_squared_error(all_targets, all_preds)
    pcc, _ = pearsonr(all_targets, all_preds)
    scc, _ = spearmanr(all_targets, all_preds)
    return mse, pcc, scc

# ====================================================================================
# 7. Main Pipeline
# ====================================================================================

def main():
    # 1. Tokenizer & Vocab
    tokenizer = AutoTokenizer.from_pretrained(YOUTU_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model_dir = Path(LOAD_MODEL_PATH)
    with open(model_dir / "args.json", "r") as f:
        model_configs = json.load(f)
    
    vocab_mod = model_configs.get("modality_dict", {'RNA': 0, 'Protein': 1, 'METHYL': 2, 'TEXT': 3, PAD_TOKEN: 4})
    special_tokens = [PAD_TOKEN, "<cls>", "<eoc>"]
    vocab_dict = {
        'RNA': GeneVocab.from_file(model_dir / "vocab_rna.json"),
        'Protein': GeneVocab.from_file(model_dir / "vocab_protein.json"),
        'METHYL': GeneVocab.from_file(model_dir / "vocab_methyl.json")
    }
    for v in vocab_dict.values():
        for s in special_tokens:
            if s not in v: v.append_token(s)

    # 2. Load & Align Data (Union)
    raw_mod_data, metadata_df, sample_ids_union = load_data_union(DATA_ROOT_DIR, DRUG_NAME, TEXT_EMBEDDING_FILE, vocab_dict)

    # 3. Create OmniFusionBlock Inputs (Masking, Comb Indices)
    omics_data, metadata_clean, final_sample_ids = create_flexmoe_input_for_cv(
        sample_ids_union, metadata_df, raw_mod_data, MODALITIES_ORDER, MODALITIES_TO_KEEP, tokenizer
    )

    # ==============================================================
    # 【新增调试代码】数据对齐实锤检查
    # ==============================================================
    print("\n" + "?"*50)
    print("[ALIGNMENT CHECK] Verifying Tensor values against Raw Data...")
    
    # 1. 随机选一个样本 ID
    check_idx = 0 
    check_id = final_sample_ids[check_idx]
    print(f"Checking Sample Index: {check_idx}, Sample ID: {check_id}")
    
    # 2. 获取该样本的 Label (Tensor vs DataFrame)
    tensor_label = metadata_clean['labels'][check_idx]
    true_label = metadata_df.loc[check_id, 'label']
    print(f"Label -> Tensor: {tensor_label:.4f} | DataFrame: {true_label:.4f}")
    
    if abs(tensor_label - true_label) > 1e-4:
        print(">>> CRITICAL ERROR: Label mismatch! Alignment is broken.")
    else:
        print(">>> Label Alignment OK.")
        
    # 3. 获取该样本的 RNA 第一个非零值 (如果存在)
    if metadata_clean['observed_idx'][check_idx][0]: # 假设 RNA 是第 0 个
        # Tensor 值
        tensor_rna_seq = omics_data['RNA_values'][check_idx]
        # 找到第一个不是 Padding 的值
        valid_indices = np.where(tensor_rna_seq != PAD_VALUE)[0]
        if len(valid_indices) > 0:
            first_val_idx = valid_indices[0]
            tensor_val = tensor_rna_seq[first_val_idx]
            gene_token_id = omics_data['RNA_genes'][check_idx][first_val_idx]
            
            # 从 Raw Data 查
            # 注意：这里需要反查 token id 对应的 gene name，比较麻烦，我们直接比对第一个值的大致范围
            print(f"RNA Tensor First Valid Value: {tensor_val:.4f}")
            
            # 你可以手动去 csv 文件里看一眼这个 ID ({check_id}) 的 RNA 数据大概是多少
            # 或者在这里加载原始 df 查一下
            # raw_val = raw_mod_data['RNA']['adata'][check_id, ...].X ...
            
    print("?"*50 + "\n")
    
    # Update metadata_df to match final samples (if filtering happened)
    metadata_df = metadata_df.loc[final_sample_ids]

    # 4. Dataset & Splitting
    full_dataset = OmniFusionBlockDataset(omics_data, metadata_clean, vocab_dict)
    
    # 这里的 split 函数会读取或创建 CSV，并执行 Val Set 的 Full 过滤
    cv_splits = create_cv_splits_after_preprocessing(metadata_df, N_SPLITS)
    
    # ID Map for dataset indexing
    id_to_dataset_idx = {sid.lower(): i for i, sid in enumerate(final_sample_ids)}

    # 5. Training Loop
    results = []
    
    for fold_info in cv_splits:
        fold = fold_info['fold']
        print(f"\n{'='*20} Fold {fold+1}/{N_SPLITS} {'='*20}")
        
        train_indices = [id_to_dataset_idx[sid.lower()] for sid in fold_info['train_ids'] if sid.lower() in id_to_dataset_idx]
        val_indices = [id_to_dataset_idx[sid.lower()] for sid in fold_info['test_ids'] if sid.lower() in id_to_dataset_idx]
        
        train_sub = torch.utils.data.Subset(full_dataset, train_indices)
        val_sub = torch.utils.data.Subset(full_dataset, val_indices)
        
        train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False)
        # --- Model Init ---
        encoder_dict = {}
        for mod in ['RNA', 'Protein', 'METHYL']:
            encoder_dict[mod] = PerformerModel(
                ntoken=len(vocab_dict[mod]),
                d_model=model_configs["layer_size"],
                nhead=model_configs["nhead"],
                d_hid=model_configs["layer_size"],
                nlayers=model_configs["nlayers"],
                vocab=vocab_dict[mod],
                pad_token=model_configs["pad_token"],
                pad_value=model_configs["pad_value"],
                n_input_bins=model_configs["n_bins"] + 2,
                cell_emb_style="cls",
                input_emb_style="category",
                dropout=DROPOUT,
            ).to(DEVICE)
        
        encoder_dict['TEXT'] = TrainableTextEncoder(
            model_name_or_path=YOUTU_PATH,
            output_dim=model_configs["layer_size"],
            dropout=DROPOUT,
            trust_remote_code=True
        ).to(DEVICE)
        text_lora_config = LoraConfig(
            r=model_configs.get("lora_rank", 16), 
            lora_alpha=model_configs.get("lora_alpha", 32), 
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
            lora_dropout=model_configs.get("dropout", 0.1), bias="none", 
            modules_to_save=["projection"]
        )
        encoder_dict['TEXT'] = get_peft_model(encoder_dict['TEXT'], text_lora_config).to(DEVICE)

        fusion_model = OmniFusionBlock(
            model_configs["num_modalities"], 0, 4,
            model_configs["layer_size"], model_configs["num_layers_fus"],
            model_configs["num_experts"], model_configs["num_routers"], 
            model_configs["top_k"], model_configs["nhead"],
            DROPOUT, device=DEVICE
        ).to(DEVICE)

        # Load Weights
        checkpoint = torch.load(model_dir / model_epoch, map_location='cpu')
        for mod in ['RNA', 'Protein', 'METHYL']:
            encoder_dict[mod].load_state_dict(checkpoint['encoder_dict'][mod], strict=True)
        encoder_dict['TEXT'].load_state_dict(checkpoint['encoder_dict']['TEXT'], strict=True)
        fusion_model.load_state_dict(checkpoint['fusion_model'], strict=True)
        # Model Wrapper (With Missing Embeds)
        model = DrugResponsePredictor(
            encoder_dict, fusion_model, model_configs["layer_size"], vocab_mod, DROPOUT, NUM_COMBINATIONS
        ).to(DEVICE)

        if freeze_text:
            print("Freezing Text Encoders")
            for p in encoder_dict['TEXT'].parameters():
                p.requires_grad = False
        if freeze_omics:
            print(f"Freezing Omic Encoders")
            for mod in ['RNA', 'Protein', 'METHYL']:
                # 1. 全量冻结 (先给大门上锁)
                for p in encoder_dict[mod].parameters():
                    p.requires_grad = False
                
        if freeze_fusion:
            print("Freezing Fusion Model (OmniFusionBlock)...")
            for p in fusion_model.parameters():
                p.requires_grad = False
            print("Freezing Missing Embeddings...")
            for p in model.missing_embeds_dict.values():
                p.requires_grad = False

        # 尝试加载预训练的 missing embeds
        print("Sample before load:", model.missing_embeds_dict['RNA'][0, 0, :5])

        model.load_missing_embeds(checkpoint, DEVICE)

        print("\n[DEBUG] Missing Embeddings Statistics:")
        for mod in MODALITIES_ORDER:
            param = model.missing_embeds_dict[mod]
            # param shape: [15, 1, 512]
            mean_val = param.data.mean().item()
            std_val = param.data.std().item()
            max_val = param.data.max().item()
            print(f"  {mod}: Mean={mean_val:.4f}, Std={std_val:.4f}, Max={max_val:.4f}")
            
            # 检查是否全 0
            if std_val == 0:
                print(f"  >>> WARNING: {mod} missing embedding has 0 variance (likely dead/not loaded).")

        # 在 load 之后
        print("Sample after load:", model.missing_embeds_dict['RNA'][0, 0, :5])
        params_to_optimize = list(filter(lambda p: p.requires_grad, model.parameters()))
        print(f"Total params: {sum(p.numel() for p in model.parameters())}")
        print(f"Trainable params (passed to Optimizer): {sum(p.numel() for p in params_to_optimize)}")

        # 优化后的验证检查 (Robust Verification)
        # ======================================================
        print("\n" + "*"*40)
        print("[Training Parameter Check]")
        
        # 统计总参数和可训练参数
        params_to_optimize = list(filter(lambda p: p.requires_grad, model.parameters()))
        trainable_num = sum(p.numel() for p in params_to_optimize)
        total_num = sum(p.numel() for p in model.parameters())
        
        print(f"Total Params: {total_num/1e6:.2f}M")
        print(f"Trainable Params: {trainable_num/1e6:.2f}M")

        def count_missing_params(model):
            total_params = 0
            print("\n[Missing Embeds Params Count]")
            for key, param in model.missing_embeds_dict.items():
                # key 是模态名 (e.g., 'RNA')
                # param shape 是 [15, 1, 512]
                count = param.numel()
                total_params += count
                print(f"  {key}: {list(param.shape)} -> {count} params")
            
            print(f"Total Missing Embeds Params: {total_params}")
            return total_params

        # 在 main 函数 model 初始化后调用
        count_missing_params(model)

        # ======================================================

        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.MSELoss()

        best_pcc = -1.0
        
        for epoch in range(EPOCHS):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, modalities_to_keep=MODALITIES_TO_KEEP)
            val_mse, val_pcc, val_scc = evaluate(model, val_loader, criterion, DEVICE, modalities_to_keep=MODALITIES_TO_KEEP)
            
            print(f"Ep {epoch+1}: Loss {train_loss:.4f} | MSE {val_mse:.4f} | PCC {val_pcc:.4f} | SCC {val_scc:.4f}")
            
            if val_pcc > best_pcc:
                best_pcc = val_pcc
                best_results = (val_mse, val_pcc, val_scc)

        results.append({'fold': fold, 'metrics': best_results})
        print(f"Fold {fold+1} Best: {best_results}")
        
        del model, optimizer, encoder_dict, fusion_model
        torch.cuda.empty_cache()

    # Summary
    metrics = np.array([r['metrics'] for r in results])
    print(f"\nFinal ({OPTION}): MSE {metrics[:,0].mean():.4f}±{metrics[:,0].std():.4f}, PCC {metrics[:,1].mean():.4f}±{metrics[:,1].std():.4f}")

if __name__ == '__main__':
    main()
