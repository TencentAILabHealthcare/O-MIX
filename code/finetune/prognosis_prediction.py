import copy
import json
import logging
import os
import sys
import warnings
from itertools import combinations
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# Hotfix for transformers
import transformers
from lifelines.utils import concordance_index
from scipy.sparse import issparse
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

if not hasattr(transformers.utils, "LossKwargs"):
    class LossKwargs: pass
    transformers.utils.LossKwargs = LossKwargs

# 添加路径
sys.path.append("../")
from omix.model.omni_fusion import \
    OmniFusionBlock
from omix.model.model_multi import PerformerModel
from omix.preprocess_bulk import Preprocessor as Preprocessor_RNA
from omix.preprocess_rppa import Preprocessor as Preprocessor_Protein
from omix.tokenizer import GeneVocab, tokenize_and_pad_batch

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ====================================================================================
# 1. 配置参数
# ====================================================================================
# 仅保留三个组学模态
OPTION = 'RPM' 
MODALITIES_ORDER = ['RNA', 'Protein', 'METHYL']
MODALITY_CHARS = {'RNA': 'R', 'Protein': 'P', 'METHYL': 'M'}

# Survival Specific Configs
DISEASE_NAME_LIST = ['OV']  # ['BRCA', 'COAD', 'LGG', 'OV']
DATA_ROOT_DIR = "../../data/GDAC"
RENAME_MAP_PATH = '../../data/gene_name_mapping/protein_name_mapping.json'
LOAD_MODEL_PATH = "../pretrain/save/omix_o_pretrain"

model_epoch = "model_e8.pt"

# Hyperparameters
freeze_omics = False
freeze_fusion = False
use_gate_loss = False
use_hvg = False

print('using modality:', OPTION)
print('freeze_omics:', freeze_omics)
print('freeze_fusion:', freeze_fusion)
print('use_hvg:', use_hvg)
print('use_gate_loss:', use_gate_loss)

gpu_index = 1
DEVICE = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(gpu_index)

BATCH_SIZE = 4 # Cox Loss benefit from larger batch size
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
EPOCHS = 50
N_SPLITS = 5
DROPOUT = 0.2
MAX_SEQ_LEN_PROTEIN = 17007 + 1
MAX_SEQ_LEN_RNA = 19205 + 1
MAX_SEQ_LEN_METHYL = 12920 + 1
PAD_TOKEN = "<pad>"
PAD_VALUE = 51
target_dir = f"survival_file_dir_novel/survival_multimodal_finetune-performer-{DISEASE_NAME_LIST[0]}-gate{use_gate_loss}-hvg{use_hvg}-DROPOUT{DROPOUT}"

print(f'freeze_omics: {freeze_omics}, freeze_fusion: {freeze_fusion}, use_hvg: {use_hvg}')
print(f'DROPOUT: {DROPOUT}, BATCH_SIZE: {BATCH_SIZE}, GRADIENT_ACCUMULATION_STEPS: {GRADIENT_ACCUMULATION_STEPS}, LEARNING_RATE:{LEARNING_RATE}')
print(f"Running Survival Prediction for {DISEASE_NAME_LIST} with OPTION: {OPTION}")

# ====================================================================================
# 2. 复用辅助函数 (From Drug Response Code)
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
print(COMBINATION_MAP)
print(f"Total Modality Combinations: {NUM_COMBINATIONS}")

def tokenize_omics(df, vocab, max_len, omics_type):
    """
    复用函数：将DataFrame转为Tokenized Tensor
    """
    adata = anndata.AnnData(df)
    adata.var['gene_name'] = adata.var.index

    if omics_type == 'METHYL':
        # RNA & Methylation logic
        pp = Preprocessor_RNA(use_key="X", filter_gene_by_counts=False, filter_cell_by_counts=False,
                            normalize_total=False, log1p=False, binning=51, result_binned_key="X_binned")
    elif omics_type == 'Protein':
        # Protein specific logic
        pp = Preprocessor_Protein(use_key="X", filter_gene_by_counts=False, filter_cell_by_counts=False,
                                normalize_total=False, log1p=False, binning=51, result_binned_key="X_binned")
    elif omics_type == 'RNA':
        # RNA & Methylation logic
        pp = Preprocessor_RNA(use_key="X", filter_gene_by_counts=False, filter_cell_by_counts=False,
                            normalize_total=False, log1p=True, binning=51, result_binned_key="X_binned")
    else:
        print('error')
        exit()
    pp(adata)
    
    # Check sparsity
    if issparse(adata.layers["X_binned"]):
        data_binned = adata.layers["X_binned"].toarray()
    else:
        data_binned = adata.layers["X_binned"]

    gene_ids = np.array(vocab(adata.var_names.tolist()), dtype=int)
    tokenized = tokenize_and_pad_batch(
        data_binned, gene_ids, max_len=max_len, 
        vocab=vocab, pad_token=PAD_TOKEN, pad_value=PAD_VALUE, append_cls=True, include_zero_gene=False
    )
    return tokenized, adata

def create_flexmoe_input_for_cv(all_sample_ids, metadata_df, modalities_data, modalities_order, modalities_to_keep, tokenizer=None, max_len_text=1024):
    """
    复用函数：灵活处理模态对齐。
    修改点：针对Survival Metadata (包含Time/Event) 进行兼容。
    """
    print("Aligning all modalities and creating dataset inputs...")

    all_sample_ids_lower = [str(sid).lower() for sid in all_sample_ids]
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
        print(f"  - Processing '{mod_name}'...")

        # 对于 omics，通常索引在 adata.obs
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

    # 2. 样本过滤
    keep_indices = [i for i, m in enumerate(modalities_order) if m in modalities_to_keep]
    valid_observed_counts = observed_idx_arr[:, keep_indices].sum(axis=1)
    valid_sample_indices = np.where(valid_observed_counts > 0)[0]
    
    if len(valid_sample_indices) < n_patients_initial:
        print(f"    [FILTERING] Dropping {n_patients_initial - len(valid_sample_indices)} samples.")
        for key in list(omics_data_dict.keys()):
            omics_data_dict[key] = omics_data_dict[key][valid_sample_indices]
        observed_idx_arr = observed_idx_arr[valid_sample_indices]
        final_sample_ids = [all_sample_ids[i] for i in valid_sample_indices]
    else:
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
        modality_comb_arr[i] = COMBINATION_MAP.get(comb_str, 0)

    # 4. 整理 Labels (Survival Specific: Time & Event)
    # metadata_df 应该包含 'OS' (Event) 和 'OS.time' (Time)
    final_sample_ids_lower = [sid.lower() for sid in final_sample_ids]
    
    # 这里我们创建临时 metadata_dict
    metadata_df.index = metadata_df.index.str.lower()
    aligned_events = metadata_df.loc[final_sample_ids_lower, 'OS'].values.astype(np.float32)
    aligned_times = metadata_df.loc[final_sample_ids_lower, 'OS.time'].values.astype(np.float32)

    metadata_dict = {
        'observed_idx': observed_idx_arr,
        'modality_comb': modality_comb_arr,
        'events': aligned_events,
        'times': aligned_times
    }
    return omics_data_dict, metadata_dict, final_sample_ids

# ====================================================================================
# 3. 适配后的 Dataset
# ====================================================================================

class OmniFusionBlockDataset(Dataset):
    """
    修改后的 Dataset，适用于 Survival Prediction
    返回: (Omics, Observed, Comb, Time, Event)
    """
    def __init__(self, omics_data, metadata, vocab_dict):
        self.omics_data = omics_data
        self.observed_idx = metadata['observed_idx']
        self.modality_comb = metadata['modality_comb']
        self.events = metadata['events']
        self.times = metadata['times']
        self.length = len(self.events)
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
                torch.tensor(self.omics_data['RNA_genes'][idx]).eq(self.vocab_dict['RNA'][PAD_TOKEN])
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
            
        item['observed'] = torch.tensor(self.observed_idx[idx], dtype=torch.bool)
        item['comb_idx'] = torch.tensor(self.modality_comb[idx], dtype=torch.long)
        
        # Survival Targets
        item['time'] = torch.tensor(self.times[idx], dtype=torch.float32)
        item['event'] = torch.tensor(self.events[idx], dtype=torch.float32)
        
        return item

# ====================================================================================
# 4. CV Splitting Logic (Survival Adapted)
# ====================================================================================

def get_stratified_kfold_splits(metadata_df, n_splits, random_state):
    """
    复用函数逻辑，但修改分层依据。
    Survival 任务通常基于 'OS' (Event状态) 进行分层。
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    sample_ids = metadata_df.index
    # 优先基于生存状态分层
    stratify_labels = metadata_df['OS'].values 
    
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
    复用函数：创建 CV 划分文件。
    """
    CV_SPLIT_FILE = Path(DATA_ROOT_DIR) / f"cv_splits_survival_{DISEASE_NAME_LIST[0]}_{n_splits}fold.csv"
    
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
        cv_splits = get_stratified_kfold_splits(metadata_df, n_splits, random_state)
        
        records = []
        for fold_info in cv_splits:
            fold = fold_info['fold']
            for sample_id in fold_info['train_ids']:
                modalities = metadata_df.loc[sample_id, 'modalities'] if sample_id in metadata_df.index else ''
                sample_type = metadata_df.loc[sample_id, 'type'] if sample_id in metadata_df.index else ''
                records.append({
                    'fold': fold, 
                    'split_type': 'train', 
                    'sample_id': sample_id,
                    'modalities': modalities,
                    'type': sample_type
                })
            for sample_id in fold_info['test_ids']:
                modalities = metadata_df.loc[sample_id, 'modalities'] if sample_id in metadata_df.index else ''
                sample_type = metadata_df.loc[sample_id, 'type'] if sample_id in metadata_df.index else ''
                records.append({
                    'fold': fold, 
                    'split_type': 'test', 
                    'sample_id': sample_id,
                    'modalities': modalities,
                    'type': sample_type
                })
        pd.DataFrame(records).to_csv(CV_SPLIT_FILE, index=False)
    
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
# 5. Data Loading (Adapted from Survival Code to match Drug Response Dict)
# ====================================================================================

def load_data_survival_union(data_dir, disease_list, vocab_dict, rename_map):
    """
    整合 Survival 的原始加载逻辑，但输出 drug response 代码需要的格式。
    """

    print("Loading raw data from GDAC...")
    all_rna_dfs, all_rppa_dfs, all_methyl_dfs = [], [], []
    sample_metadata_rows = []

    for disease in disease_list:
        disease_path = os.path.join(data_dir, disease)
        try:
            labels_df = pd.read_csv(os.path.join(disease_path, 'labels.csv'), sep='\t', index_col=0)
            mrna = pd.read_csv(os.path.join(disease_path, 'mRNA_TPM.csv'), index_col=0)
            methyl = pd.read_csv(os.path.join(disease_path, 'methylation.csv'), index_col=0)
            protein = pd.read_csv(os.path.join(disease_path, 'RPPA.csv'), index_col=0)
        except Exception as e:
            print(f"Skipping {disease}: {e}")
            continue

        # Standardize Indices
        def clean_idx(df):
            if df.empty: return df
            df.index = df.index.str.slice(0, 12).str.lower()
            return df[~df.index.duplicated(keep='first')]

        labels_df = clean_idx(labels_df)
        mrna = clean_idx(mrna)
        methyl = clean_idx(methyl)
        protein = clean_idx(protein)

        # Filter valid survival labels
        labels_df = labels_df[(labels_df['Time'] > 0) & (labels_df.index.isin(
            set(mrna.index) | set(methyl.index) | set(protein.index)
        ))]

        if not protein.empty:
            protein.columns = [x.lower() for x in protein.columns]
            protein = protein.rename(columns=rename_map)
            protein = protein.loc[:, ~protein.columns.duplicated()]

        all_rna_dfs.append(mrna)
        all_rppa_dfs.append(protein)
        all_methyl_dfs.append(methyl)

        # Build Metadata
        for sid in labels_df.index:
            mods = []
            if sid in mrna.index: mods.append('RNA')
            if sid in protein.index: mods.append('Protein')
            if sid in methyl.index: mods.append('METHYL')
            
            sample_metadata_rows.append({
                'sample_id': sid,
                'OS': labels_df.loc[sid, 'Event'],
                'OS.time': labels_df.loc[sid, 'Time'],
                'modalities': ",".join(mods),
                'type': 'full' if len(mods) == 3 else 'partial'
            })

    # Combine DataFrames
    def combine_dfs(dfs):
        if not dfs: return pd.DataFrame()
        return pd.concat(dfs).groupby(level=0).first()

    df_rna = combine_dfs(all_rna_dfs)
    df_prot = combine_dfs(all_rppa_dfs)
    df_meth = combine_dfs(all_methyl_dfs)
    
    metadata_df = pd.DataFrame(sample_metadata_rows).set_index('sample_id')
    metadata_df = metadata_df[~metadata_df.index.duplicated(keep='first')]

    # Prepare Dictionary for tokenize_omics
    modalities_data = {}
    
    # RNA
    if not df_rna.empty:
        tok_rna, adata_rna = tokenize_omics(df_rna, vocab_dict['RNA'], MAX_SEQ_LEN_RNA, 'RNA')
        modalities_data['RNA'] = {'type': 'omics', 'adata': adata_rna, 'tokenized': tok_rna, 'vocab': vocab_dict['RNA'], 'char': 'R'}
    
    # Protein
    if not df_prot.empty:
        tok_prot, adata_prot = tokenize_omics(df_prot, vocab_dict['Protein'], MAX_SEQ_LEN_PROTEIN, 'Protein')
        modalities_data['Protein'] = {'type': 'omics', 'adata': adata_prot, 'tokenized': tok_prot, 'vocab': vocab_dict['Protein'], 'char': 'P'}

    # Methyl
    if not df_meth.empty:
        tok_meth, adata_meth = tokenize_omics(df_meth, vocab_dict['METHYL'], MAX_SEQ_LEN_METHYL, 'METHYL')
        modalities_data['METHYL'] = {'type': 'omics', 'adata': adata_meth, 'tokenized': tok_meth, 'vocab': vocab_dict['METHYL'], 'char': 'M'}

    return modalities_data, metadata_df, metadata_df.index.tolist()

# ====================================================================================
# 6. Model Definition (SurvivalPredictor)
# ====================================================================================

class SurvivalPredictor(nn.Module):
    """
    重构自 DrugResponsePredictor。
    主要区别：Head 输出单值 (Log Hazard)，无 Text Encoder。
    """
    def __init__(self, encoder_dict, fusion_model, embed_dim, vocab_mod_map, dropout, num_combinations):
        super().__init__()
        # self.encoder_dict = encoder_dict
        self.encoder_dict = nn.ModuleDict(encoder_dict)
        self.fusion_model = fusion_model
        self.vocab_mod_map = vocab_mod_map
        self.embed_dim = embed_dim
        
        self.missing_embeds_dict = nn.ParameterDict()
        for mod in MODALITIES_ORDER:
            self.missing_embeds_dict[mod] = nn.Parameter(torch.randn(num_combinations, 1, embed_dim))

        # Survival Head: Output 1 scalar (log hazard ratio)
        head_in_dim = embed_dim * len(MODALITIES_ORDER)
        self.head = nn.Sequential(
            nn.Linear(head_in_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1) 
        )

    def load_missing_embeds(self, checkpoint, device):
        if 'missing_embeds' not in checkpoint: 
            print('missing embeds not in checkpoint')
            return
        pretrained_missing = checkpoint['missing_embeds']
        for mod_name, param in self.missing_embeds_dict.items():
            if mod_name in pretrained_missing:
                full_tensor = pretrained_missing[mod_name].to(device)
                mod_idx = self.vocab_mod_map.get(mod_name)
                sliced_tensor = full_tensor[:, mod_idx, :, :]
                if sliced_tensor.dim() == 2: sliced_tensor = sliced_tensor.unsqueeze(1)
                if sliced_tensor.shape == param.shape:
                    param.data.copy_(sliced_tensor)
        print('loading missing_embeds finished')

    def run_encoder(self, mod_name, batch_data, device):
        genes, vals, mask = batch_data
        genes, vals, mask = genes.to(device), vals.to(device), mask.to(device)
        enc_out, _ = self.encoder_dict[mod_name](
            src=genes, values=vals, src_key_padding_mask=mask, return_seq_embedding=True
        )
        return enc_out[:, 0:1, :] # CLS token

    def run_adapter_safe(self, mod_name, feat):
        if hasattr(self.fusion_model, "module"):
            return self.fusion_model.module.run_adapter(mod_name, feat)
        return self.fusion_model.run_adapter(mod_name, feat)

    def forward(self, batch_dict):
        device = next(self.parameters()).device
        observed_mask = batch_dict['observed'].to(device)
        comb_idx = batch_dict['comb_idx'].to(device)
        batch_size = observed_mask.shape[0]

        embeddings_list = []
        mod_types_list = []
        
        for i, mod in enumerate(MODALITIES_ORDER):
            feat_tensor = torch.zeros(batch_size, 1, self.embed_dim, device=device)
            present_indices = torch.where(observed_mask[:, i])[0]
            missing_indices = torch.where(~observed_mask[:, i])[0]
            
            # Present
            if len(present_indices) > 0:
                sub_data_cpu = batch_dict[mod]
                sub_data_gpu = [t.to(device) for t in sub_data_cpu]
                sub_inputs = (sub_data_gpu[0][present_indices], sub_data_gpu[1][present_indices], sub_data_gpu[2][present_indices])
                raw_feat = self.run_encoder(mod, sub_inputs, device)
                feat_tensor[present_indices] = raw_feat

            # Missing
            if len(missing_indices) > 0:
                missing_combs = comb_idx[missing_indices]
                miss_feat = self.missing_embeds_dict[mod][missing_combs]
                feat_tensor[missing_indices] = miss_feat
            
            feat_tensor = self.run_adapter_safe(mod, feat_tensor)
            embeddings_list.append(feat_tensor)
            mod_id = self.vocab_mod_map[mod]
            mod_types_list.append(torch.full((batch_size, 1), mod_id, dtype=torch.long, device=device))

        mod_types = torch.cat(mod_types_list, dim=1)
        final_repr = self.fusion_model(
            *embeddings_list, expert_indices=comb_idx, mod_types=mod_types, return_cell_embedding=True
        )
        final_repr = final_repr.reshape(batch_size, -1)
        
        # Output Log Hazard
        risk_score = self.head(final_repr).squeeze(-1)

        gate_loss = self.fusion_model.gate_loss()
        return risk_score, gate_loss

class CoxPHLoss(nn.Module):
    def forward(self, risks, times, events):
        # risks: (B,)
        # times: (B,)
        # events: (B,)
        durations, sort_idx = torch.sort(times, descending=True)
        events = events[sort_idx]
        log_hazards = risks[sort_idx]
        log_risk_sum = torch.logcumsumexp(log_hazards, dim=0)
        loss = -torch.sum(log_hazards[events.bool()] - log_risk_sum[events.bool()])
        num_events = torch.sum(events)
        return loss / num_events if num_events > 0 else torch.tensor(0.0, requires_grad=True, device=risks.device)

# ====================================================================================
# 7. Training Functions
# ====================================================================================

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    for i, batch in enumerate(tqdm(dataloader, desc="Training")):
        times = batch['time'].to(device)
        events = batch['event'].to(device)
        
        preds, gate_loss = model(batch) # Returns risk scores
        
        loss = criterion(preds, times, events)
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
        del loss, preds, gate_loss, times, events
    
    # ✅ 修复：处理未完整累积的梯度
    if len(dataloader) % GRADIENT_ACCUMULATION_STEPS != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    all_risks = []
    all_times = []
    all_events = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            times = batch['time']
            events = batch['event']
            preds, _ = model(batch)
            
            all_risks.append(preds.cpu().numpy())
            all_times.append(times.cpu().numpy())
            all_events.append(events.cpu().numpy())
            
            # 及时释放显存
            del preds, times, events, batch
    
    all_risks = np.concatenate(all_risks)
    all_times = np.concatenate(all_times)
    all_events = np.concatenate(all_events)
    
    try:
        # C-Index: measures concordance between risk scores and time-to-event
        # Note: Higher risk should correlate with lower time, lifelines expects -risk for standard check
        # But CoxPH outputs log(hazard). Higher hazard = shorter time.
        # c_index(time, predicted_risk, event) usually expects predicted risk to be high for short survival.
        # Let's use lifelines standard:
        c_idx = concordance_index(all_times, -all_risks, all_events)
    except Exception as e:
        print(f"C-Index Error: {e}")
        c_idx = 0.5
        
    return c_idx, all_times, all_events, all_risks

# ====================================================================================
# 8. Main
# ====================================================================================

def main():
    output_dir = Path(target_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # 1. Config & Vocab
    model_dir = Path(LOAD_MODEL_PATH)
    with open(model_dir / "args.json", "r") as f:
        model_configs = json.load(f)
    
    vocab_mod = model_configs.get("modality_dict", {'RNA': 0, 'Protein': 1, 'METHYL': 2, PAD_TOKEN: 4})
    if PAD_TOKEN not in vocab_mod:
        vocab_mod[PAD_TOKEN] = len(vocab_mod)
    special_tokens = [PAD_TOKEN, "<cls>", "<eoc>"]
    vocab_dict = {
        'RNA': GeneVocab.from_file(model_dir / "vocab_rna.json"),
        'Protein': GeneVocab.from_file(model_dir / "vocab_protein.json"),
        'METHYL': GeneVocab.from_file(model_dir / "vocab_methyl.json")
    }

    for v in vocab_dict.values():
        for s in special_tokens:
            if s not in v: v.append_token(s)

    with open(RENAME_MAP_PATH, 'r') as f:
        rename_map = json.load(f)

    # 2. Load Data (Survival Style -> OmniFusionBlock Style)
    raw_mod_data, metadata_df, sample_ids_union = load_data_survival_union(
        DATA_ROOT_DIR, DISEASE_NAME_LIST, vocab_dict, rename_map
    )

    # 3. Create Inputs
    omics_data, metadata_clean, final_sample_ids = create_flexmoe_input_for_cv(
        sample_ids_union, metadata_df, raw_mod_data, MODALITIES_ORDER, MODALITIES_ORDER
    )
    
    # Update DF alignment
    metadata_df = metadata_df.loc[final_sample_ids]

    # 4. Dataset
    full_dataset = OmniFusionBlockDataset(omics_data, metadata_clean, vocab_dict)
    
    # 5. Splitting (Stratified by Event)
    id_to_dataset_idx = {sid.lower(): i for i, sid in enumerate(final_sample_ids)}
    cv_splits = create_cv_splits_after_preprocessing(metadata_df, N_SPLITS) # This creates CSV
    
    # 6. Loop
    results = []
    all_folds_predictions = []
    criterion = CoxPHLoss()
    
    for fold_info in cv_splits:
        fold = fold_info['fold']
        # 创建当前fold的输出目录
        fold_dir = output_dir / f"fold_{fold + 1}"
        fold_dir.mkdir(exist_ok=True)
        print(f"\n{'='*20} Fold {fold+1}/{N_SPLITS} {'='*20}")
        
        train_indices = [id_to_dataset_idx[str(sid).lower()] for sid in fold_info['train_ids'] if str(sid).lower() in id_to_dataset_idx]
        val_indices = [id_to_dataset_idx[str(sid).lower()] for sid in fold_info['test_ids'] if str(sid).lower() in id_to_dataset_idx]

        print(f"Saving train_idx, test_idx and processed data for fold {fold + 1}...")
        
        # 1. 保存索引 (转为 numpy 格式保存)
        np.save(fold_dir / "train_idx.npy", np.array(train_indices))
        np.save(fold_dir / "test_idx.npy", np.array(val_indices))
        
        # 也可以顺便保存原始的 Sample ID 列表 (文本格式)，方便肉眼核对
        pd.Series(fold_info['train_ids']).to_csv(fold_dir / "train_ids.csv", index=False)
        pd.Series(fold_info['test_ids']).to_csv(fold_dir / "test_ids.csv", index=False)

        # 2. 保存处理后的数据
        # 从 full_dataset 中提取 numpy/tensor 数据
        processed_data_to_save = {
            'times': full_dataset.times,                 # 对应之前的 durations
            'events': full_dataset.events,               # 对应之前的 events
            'omics_data': full_dataset.omics_data,       # 对应之前的 data_sources (输入特征)
            'observed_idx': full_dataset.observed_idx,   # 模态观测掩码
            'modality_comb': full_dataset.modality_comb, # 组合索引
            'vocab_dict': full_dataset.vocab_dict        # 词表，复现时很有用
        }
        
        # 使用 torch.save 保存字典
        torch.save(processed_data_to_save, fold_dir / "processed_data.pt")
        print(f"Saved processed data for fold {fold + 1}.")
        
        use_weighted = True  # 开关
        train_sampler = None
        
        if use_weighted:
            print("Creating WeightedRandomSampler for the training set...")
            
            # 1. 获取当前训练集的事件状态
            # full_dataset.events 是 numpy 数组，可以直接索引
            # 确保转为 tensor 以便进行 torch.where 和 sum 操作
            events_train = torch.tensor(full_dataset.events[train_indices], dtype=torch.float32)
            
            n_samples_train = len(events_train)
            n_events_train = torch.sum(events_train).item()
            n_censored_train = n_samples_train - n_events_train

            if n_events_train > 0 and n_censored_train > 0:
                # 2. 计算权重：数量少的类别权重高
                weight_event = n_samples_train / n_events_train
                weight_censored = n_samples_train / n_censored_train
                
                # 3. 为每个样本分配权重
                sample_weights = torch.where(events_train == 1, weight_event, weight_censored)

                # 4. 创建采样器
                train_sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True # 有放回采样，这对平衡类别至关重要
                )
                print(f"Sampler created. Event weight: {weight_event:.2f}, Censored weight: {weight_censored:.2f}")
            else:
                print("Training set has only one class or invalid counts. Sampler disabled for this fold.")

        # ==============================================================================
        
        train_sub = torch.utils.data.Subset(full_dataset, train_indices)
        val_sub = torch.utils.data.Subset(full_dataset, val_indices)
        
        # [修改] DataLoader 初始化
        if use_weighted and train_sampler is not None:
            # 注意: 使用 sampler 时，shuffle 必须为 False
            train_loader = DataLoader(
                train_sub, 
                batch_size=BATCH_SIZE, 
                sampler=train_sampler, 
                shuffle=False, # 互斥
                drop_last=True 
            )
        else:
            train_loader = DataLoader(
                train_sub, 
                batch_size=BATCH_SIZE, 
                shuffle=True, 
                drop_last=True
            )
            
        val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False)

        # Model Init
        encoder_dict = {}
        for mod in MODALITIES_ORDER:
            encoder_dict[mod] = PerformerModel(
                ntoken=len(vocab_dict[mod]),
                d_model=model_configs["layer_size"], nhead=model_configs["nhead"],
                d_hid=model_configs["layer_size"], nlayers=model_configs["nlayers"],
                vocab=vocab_dict[mod], pad_token=model_configs["pad_token"], pad_value=model_configs["pad_value"],
                n_input_bins=model_configs["n_bins"] + 2, cell_emb_style="cls", input_emb_style="category",
                dropout=DROPOUT
            ).to(DEVICE)

        fusion_model = OmniFusionBlock(
            model_configs["num_modalities"], 0, 3,
            model_configs["layer_size"], model_configs["num_layers_fus"],
            model_configs["num_experts"], model_configs["num_routers"], 
            model_configs["top_k"], model_configs["nhead"],
            DROPOUT, vocab_mod=vocab_mod, device=DEVICE
        ).to(DEVICE)

        # Load Weights
        checkpoint = torch.load(model_dir / model_epoch, map_location='cpu')
        for mod in MODALITIES_ORDER:
            encoder_dict[mod].load_state_dict(checkpoint['encoder_dict'][mod], strict=True)
        fusion_model.load_state_dict(checkpoint['fusion_model'], strict=True)
        
        model = SurvivalPredictor(
            encoder_dict, fusion_model, model_configs["layer_size"], vocab_mod, DROPOUT, NUM_COMBINATIONS
        ).to(DEVICE)

        # Freeze Logic
        if freeze_omics:
            for mod in MODALITIES_ORDER:
                for p in encoder_dict[mod].parameters(): p.requires_grad = False
        if freeze_fusion:
            for p in fusion_model.parameters(): p.requires_grad = False
            
        model.load_missing_embeds(checkpoint, DEVICE)
        
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        best_c_index = 0.0
        best_fold_pred_data = None
        for epoch in range(EPOCHS):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
            val_c_index, val_times, val_events, val_risks = evaluate(model, val_loader, DEVICE)
            torch.cuda.empty_cache() 
            print(f"Ep {epoch+1}: Train Loss {train_loss:.4f} | C-Index {val_c_index:.4f}")
            if val_c_index > best_c_index:
                best_c_index = val_c_index
                # best_model_state = copy.deepcopy(model.state_dict())
                print(f"  >>> New best model for this fold saved with Test C-index: {best_c_index:.4f} <<<")


                best_model_state = {
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'c_index': best_c_index
                }

                best_fold_pred_data = {
                    'Fold': fold,
                    'Time': val_times.copy(),
                    'Event': val_events.copy(),
                    'Risk_Score': val_risks.copy()
                }
                
                # 定义保存路径并保存
                # save_path = fold_dir / "best_model_allgenes.pt"
                save_path = fold_dir / "best_model.pt"
                torch.save(best_model_state, save_path)
                
                print(f"  >>> New best model for fold {fold + 1} saved to {save_path} with Test C-index: {best_c_index:.4f} <<<")

        # --- d. 记录该折的最佳结果 ---
        print(f"\n--- Fold {fold + 1} Finished. Best Test C-index: {best_c_index:.4f} ---")
        results.append(best_c_index)
        if best_fold_pred_data is not None:
            # 转换为 DataFrame
            fold_df = pd.DataFrame({
                'Fold': [best_fold_pred_data['Fold']] * len(best_fold_pred_data['Time']),
                'Time': best_fold_pred_data['Time'],
                'Event': best_fold_pred_data['Event'],
                'Risk_Score': best_fold_pred_data['Risk_Score']
            })
            all_folds_predictions.append(fold_df)

    # --- 5. 交叉验证结束后，报告最终聚合结果 ---
    print(f"\n{'=' * 25} FINAL CROSS-VALIDATION SUMMARY {'=' * 25}")
    mean_c_index = np.mean(results)
    std_c_index = np.std(results)

    print(f"C-index over {N_SPLITS} folds: {results}")
    print(f"Average Test C-index: {mean_c_index:.4f} ± {std_c_index:.4f}")

    # 保存最终结果
    save_results = {
        "disease": DISEASE_NAME_LIST[0],
        "n_splits": N_SPLITS,
        "mean_test_c_index": mean_c_index,
        "std_test_c_index": std_c_index,
        "individual_fold_c_indices": results,
        "epochs_per_fold": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE
    }
    with open(output_dir / "final_results.json", 'w') as f:
        json.dump(save_results, f, indent=4)

    if all_folds_predictions:
        final_df = pd.concat(all_folds_predictions, ignore_index=True)
        model_suffix = os.path.basename(os.path.normpath(LOAD_MODEL_PATH))
        save_path = output_dir / f"predictions_for_plots_{DISEASE_NAME_LIST[0]}_{model_suffix}_hvg{use_hvg}.csv"
        final_df.to_csv(save_path, index=False)
        print(f"Detailed predictions saved to: {save_path}")
        print(f"Columns: Fold, Time, Event, Risk_Score")

    print(f"\nTraining finished. All results saved to {output_dir}")
        

if __name__ == '__main__':
    main()
