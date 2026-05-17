import copy
import json
import logging
import os
import sys
import warnings
from itertools import combinations
from pathlib import Path

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
# Hotfix for transformers
import transformers
from scipy.sparse import issparse
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, label_binarize
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

if not hasattr(transformers.utils, "LossKwargs"):
    class LossKwargs: pass
    transformers.utils.LossKwargs = LossKwargs

# Add path
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
# 1. Configuration
# ====================================================================================
# Modality Configs
OPTION = 'RPM' 
MODALITIES_ORDER = ['RNA', 'Protein', 'METHYL']
MODALITY_CHARS = {'RNA': 'R', 'Protein': 'P', 'METHYL': 'M'}

# Classification Specific Configs
DISEASE_NAME_LIST = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC', 'KICH',
                     'KIRC', 'KIRP',  'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'READ', 'SARC', 'SKCM',
                     'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS']
# For debugging, you can uncomment below:
# DISEASE_NAME_LIST = ['ACC', 'BLCA', 'BRCA'] 

DATA_ROOT_DIR = "../../data/GDAC"
RENAME_MAP_PATH = '../../data/gene_name_mapping/protein_name_mapping.json'
LOAD_MODEL_PATH = "../pretrain/save/omix_o_pretrain"


model_epoch = "model_e8.pt"

# Hyperparameters
freeze_omics = False
freeze_fusion = False
use_hvg = False
use_gate_loss = False
gpu_index = 1
DEVICE = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(gpu_index)
print('using modality:', OPTION)
print('freeze_omics:', freeze_omics)
print('freeze_fusion:', freeze_fusion)
print('use_hvg:', use_hvg)
print('use_gate_loss:', use_gate_loss)
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
EPOCHS = 100
N_SPLITS = 5
DROPOUT = 0.2
MAX_SEQ_LEN_PROTEIN = 17007 + 1
MAX_SEQ_LEN_RNA = 19205 + 1
MAX_SEQ_LEN_METHYL = 12920 + 1
PAD_TOKEN = "<pad>"
PAD_VALUE = 51

target_dir = f"classification_file_dir_novel/classification-LEARNING_RATE{LEARNING_RATE}-DROPOUT{DROPOUT}-BATCH_SIZE{BATCH_SIZE}-GRADIENT_ACCUMULATION_STEPS{GRADIENT_ACCUMULATION_STEPS}_hvg{use_hvg}_gateloss{use_gate_loss}"
print(f'freeze_omics: {freeze_omics}, freeze_fusion: {freeze_fusion}, use_hvg: {use_hvg}')
print(f'DROPOUT: {DROPOUT}, BATCH_SIZE: {BATCH_SIZE}, GRADIENT_ACCUMULATION_STEPS: {GRADIENT_ACCUMULATION_STEPS}, LEARNING_RATE:{LEARNING_RATE}')
print(f"Running Classification for {len(DISEASE_NAME_LIST)} diseases with OPTION: {OPTION}")

# ====================================================================================
# 2. Helper Functions (Reused from Survival Code Structure)
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
    Transforms DataFrame to Tokenized Tensor using specific Preprocessors.
    """
    adata = anndata.AnnData(df)
    adata.var['gene_name'] = adata.var.index

    if omics_type == 'METHYL':
        pp = Preprocessor_RNA(use_key="X", filter_gene_by_counts=False, filter_cell_by_counts=False,
                            normalize_total=False, log1p=False, binning=51, result_binned_key="X_binned", repair=False)
    elif omics_type == 'Protein':
        # Specific RPPA preprocessing from Code 2
        pp = Preprocessor_Protein(
            use_key="X", filter_gene_by_counts=False, filter_cell_by_counts=False,
            normalize_total=False, result_normed_key="X_normed", log1p=False, 
            result_log1p_key="X_log1p", subset_hvg=False, hvg_flavor="seurat", 
            binning=51, result_binned_key="X_binned"
        )
    elif omics_type == 'RNA':
        pp = Preprocessor_RNA(use_key="X", filter_gene_by_counts=False, filter_cell_by_counts=False,
                            normalize_total=False, log1p=True, binning=51, result_binned_key="X_binned", repair=False)
    else:
        raise ValueError(f"Unknown omics type: {omics_type}")
    
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

def create_flexmoe_input_for_cv(all_sample_ids, metadata_df, modalities_data, modalities_order, modalities_to_keep):
    """
    Aligns modalities and prepares input dicts.
    Modified for Classification: Handles 'labels' instead of 'events'/'times'.
    """
    print("Aligning all modalities and creating dataset inputs...")

    all_sample_ids_lower = [str(sid).lower() for sid in all_sample_ids]
    id_to_idx = {sid: i for i, sid in enumerate(all_sample_ids_lower)}
    n_patients_initial = len(all_sample_ids)

    omics_data_dict = {}
    modality_to_order_idx = {name: i for i, name in enumerate(modalities_order)}
    observed_idx_arr = np.zeros((n_patients_initial, len(modalities_order)), dtype=bool)
    
    # 1. Fill Data
    for mod_name, mod_data in modalities_data.items():
        if mod_name not in modalities_order: continue
        
        mod_order_idx = modality_to_order_idx[mod_name]
        print(f"  - Processing '{mod_name}'...")

        ptids_in_modality = [str(pid).lower() for pid in mod_data['adata'].obs.index.tolist()]
        source_lookup = {str(pid).lower(): i for i, pid in enumerate(mod_data['adata'].obs.index)}

        global_indices = [id_to_idx[ptid] for ptid in ptids_in_modality if ptid in id_to_idx]
        if not global_indices: continue
        
        observed_idx_arr[global_indices, mod_order_idx] = True
        valid_ptids_lower = [all_sample_ids_lower[i] for i in global_indices]
        source_indices = [source_lookup[pid] for pid in valid_ptids_lower]

        tokenized_data = mod_data['tokenized']
        vocab = mod_data['vocab']
        
        genes_full = np.full((n_patients_initial, tokenized_data['genes'].shape[1]), vocab[PAD_TOKEN], dtype=np.int64)
        values_full = np.full((n_patients_initial, tokenized_data['values'].shape[1]), PAD_VALUE, dtype=np.float32)
        
        genes_full[global_indices] = tokenized_data['genes'][source_indices]
        values_full[global_indices] = tokenized_data['values'][source_indices]
        
        omics_data_dict[f'{mod_name}_genes'] = genes_full
        omics_data_dict[f'{mod_name}_values'] = values_full

    # 2. Filter Samples
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

    # 3. Generate Modality Combination Index
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

    # 4. Prepare Labels (Classification Specific)
    final_sample_ids_lower = [sid.lower() for sid in final_sample_ids]
    
    # Ensure metadata index is lower case for matching
    metadata_df.index = metadata_df.index.str.lower()
    aligned_labels = metadata_df.loc[final_sample_ids_lower, 'label_encoded'].values.astype(np.int64)

    metadata_dict = {
        'observed_idx': observed_idx_arr,
        'modality_comb': modality_comb_arr,
        'labels': aligned_labels
    }
    return omics_data_dict, metadata_dict, final_sample_ids

# ====================================================================================
# 3. Dataset & Split Logic
# ====================================================================================

class OmniFusionBlockDataset(Dataset):
    """
    Refactored Dataset for Classification.
    Returns: (Omics, Observed, Comb, Label)
    """
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
        item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item

def get_stratified_kfold_splits(metadata_df, n_splits, random_state):
    """
    Stratified based on 'type' (Full vs Partial modalities) to ensure balanced evaluation,
    or can stratify based on 'pancancer_type' (Disease label).
    Here we stratify based on Disease Type to ensure all folds have all classes.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    sample_ids = metadata_df.index
    # Stratify by Disease Label
    stratify_labels = metadata_df['pancancer_type'].values 
    
    splits = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(sample_ids, stratify_labels)):
        splits.append({
            'fold': fold, 
            'train_ids': sample_ids[train_idx].tolist(), 
            'test_ids': sample_ids[test_idx].tolist()
        })
    return splits

def create_cv_splits_after_preprocessing(metadata_df, n_splits=5, random_state=42, filter_val_for_common=True):
    CV_SPLIT_FILE = Path(DATA_ROOT_DIR) / f"cv_splits_classification_{n_splits}fold.csv"
    
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
# 4. Data Loading
# ====================================================================================

def load_data_classification(data_dir, disease_list, vocab_dict, rename_map):
    """
    Loads data for all diseases.
    """

    print("Loading raw data from GDAC...")
    all_rna_dfs, all_rppa_dfs, all_methyl_dfs = [], [], []
    sample_metadata_rows = []

    for disease in disease_list:
        print('loading disease:', disease)
        disease_path = os.path.join(data_dir, disease)
        if not os.path.exists(disease_path): continue

        try:
            # Code 2 style loading
            try: mrna = pd.read_csv(os.path.join(disease_path, 'mRNA_TPM.csv'), index_col=0)
            except: mrna = pd.DataFrame()
            try: methyl = pd.read_csv(os.path.join(disease_path, 'methylation.csv'), index_col=0)
            except: methyl = pd.DataFrame()
            try: protein = pd.read_csv(os.path.join(disease_path, 'RPPA.csv'), index_col=0)
            except: protein = pd.DataFrame()
        except Exception as e:
            print(f"Skipping {disease}: {e}")
            continue

        # Standardize Indices
        def clean_idx(df):
            if df.empty: return df
            df.index = df.index.str.slice(0, 12).str.lower()
            return df[~df.index.duplicated(keep='first')]

        mrna = clean_idx(mrna)
        methyl = clean_idx(methyl)
        protein = clean_idx(protein)

        if not protein.empty:
            protein.columns = [x.lower() for x in protein.columns]
            protein = protein.rename(columns=rename_map)
            protein = protein.loc[:, ~protein.columns.duplicated()]

        # Identify samples
        current_samples = set(mrna.index) | set(protein.index) | set(methyl.index)
        for sid in current_samples:
            mods = []
            if sid in mrna.index: mods.append('RNA')
            if sid in protein.index: mods.append('Protein')
            if sid in methyl.index: mods.append('METHYL')
            if disease == 'COAD' or disease == 'READ':
                disease_name = 'COADREAD'
            else:
                disease_name = disease
            sample_metadata_rows.append({
                'sample_id': sid,
                'pancancer_type': disease_name, # Label
                'modalities': ",".join(mods),
                'type': 'full' if len(mods) == 3 else 'partial'
            })
            
        if not mrna.empty: all_rna_dfs.append(mrna)
        if not protein.empty: all_rppa_dfs.append(protein)
        if not methyl.empty: all_methyl_dfs.append(methyl)

    # Combine DataFrames
    def combine_dfs(dfs):
        if not dfs: return pd.DataFrame()
        return pd.concat(dfs).groupby(level=0).first()

    df_rna = combine_dfs(all_rna_dfs)
    df_prot = combine_dfs(all_rppa_dfs)
    df_meth = combine_dfs(all_methyl_dfs)
    
    metadata_df = pd.DataFrame(sample_metadata_rows).set_index('sample_id')
    metadata_df = metadata_df[~metadata_df.index.duplicated(keep='first')]
    
    print(f"Total Combined Samples: {len(metadata_df)}")

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
# 5. Model Definition
# ====================================================================================

class ClassificationPredictor(nn.Module):
    """
    Refactored from SurvivalPredictor. 
    Head outputs num_classes.
    """
    def __init__(self, encoder_dict, fusion_model, embed_dim, vocab_mod_map, dropout, num_combinations, num_classes):
        super().__init__()
        # self.encoder_dict = encoder_dict
        self.encoder_dict = nn.ModuleDict(encoder_dict)
        self.fusion_model = fusion_model
        self.vocab_mod_map = vocab_mod_map
        self.embed_dim = embed_dim
        
        self.missing_embeds_dict = nn.ParameterDict()
        for mod in MODALITIES_ORDER:
            self.missing_embeds_dict[mod] = nn.Parameter(torch.randn(num_combinations, 1, embed_dim))

        # Classification Head: Output num_classes
        head_in_dim = embed_dim * len(MODALITIES_ORDER)
        self.head = nn.Sequential(
            nn.Linear(head_in_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes) 
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
                # Broadcasat handling from Code 2
                elif sliced_tensor.shape[1] == 1 and param.shape[1] != 1:
                     param.data.copy_(sliced_tensor.expand_as(param))
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
        
        logits = self.head(final_repr)
        gate_loss = self.fusion_model.gate_loss()
        return logits, gate_loss

# ====================================================================================
# 6. Training & Eval
# ====================================================================================

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    for i, batch in enumerate(tqdm(dataloader, desc="Training")):
        labels = batch['label'].to(device)
        
        logits, gate_loss = model(batch)
        
        loss = criterion(logits, labels)
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
    
    if len(dataloader) % GRADIENT_ACCUMULATION_STEPS != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            labels = batch['label']
            logits, _ = model(batch)
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            del logits, probs, preds, labels, batch

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def calculate_all_metrics(labels, preds, probs, n_classes):
    unique_classes = np.unique(labels)
    n_unique_classes = len(unique_classes)

    if n_classes == 2:
        # Binary case handling if num_classes in dataset is 2
        # Assuming probs has shape [N, 2]
        pos_idx = 1
        auc_score_val = roc_auc_score(labels, probs[:, pos_idx])
        aupr_score_val = average_precision_score(labels, probs[:, pos_idx], average='weighted')
    else:
        # Multiclass
        try:
            auc_score_val = roc_auc_score(labels, probs, multi_class='ovr', average='weighted')
        except:
            auc_score_val = 0.5
        
        labels_one_hot = label_binarize(labels, classes=range(n_classes))
        try:
            aupr_score_val = average_precision_score(labels_one_hot, probs, average='weighted')
        except:
            aupr_score_val = 0.0

    metrics = {
        'ACC': accuracy_score(labels, preds),
        'F1': f1_score(labels, preds, average='weighted', zero_division=0),
        'Recall': recall_score(labels, preds, average='weighted', zero_division=0),
        'Precision': precision_score(labels, preds, average='weighted', zero_division=0),
        'AUPR': aupr_score_val,
        'AUC': auc_score_val
    }
    return metrics

# ====================================================================================
# 7. Main
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

    # 2. Load Data
    raw_mod_data, metadata_df, sample_ids_union = load_data_classification(
        DATA_ROOT_DIR, DISEASE_NAME_LIST, vocab_dict, rename_map
    )
    
    # Encode Labels
    le = LabelEncoder()
    metadata_df['label_encoded'] = le.fit_transform(metadata_df['pancancer_type'])
    n_classes = len(le.classes_)
    print(f"Classes found: {n_classes} -> {le.classes_}")

    # 3. Create Inputs
    omics_data, metadata_clean, final_sample_ids = create_flexmoe_input_for_cv(
        sample_ids_union, metadata_df, raw_mod_data, MODALITIES_ORDER, MODALITIES_ORDER
    )
    
    # Update Metadata Alignment
    metadata_df = metadata_df.loc[final_sample_ids]

    # 4. Dataset
    full_dataset = OmniFusionBlockDataset(omics_data, metadata_clean, vocab_dict)
    
    # 5. Splitting
    id_to_dataset_idx = {sid.lower(): i for i, sid in enumerate(final_sample_ids)}
    cv_splits = create_cv_splits_after_preprocessing(metadata_df, N_SPLITS)
    
    # 6. Loop
    test_fold_metrics = []
    all_folds_labels, all_folds_preds = [], []
    criterion = nn.CrossEntropyLoss()
    
    for fold_info in cv_splits:
        fold = fold_info['fold']
        fold_dir = output_dir / f"fold_{fold + 1}"
        fold_dir.mkdir(exist_ok=True)
        print(f"\n{'='*20} Fold {fold+1}/{N_SPLITS} {'='*20}")
        
        train_indices = [id_to_dataset_idx[str(sid).lower()] for sid in fold_info['train_ids'] if str(sid).lower() in id_to_dataset_idx]
        val_indices = [id_to_dataset_idx[str(sid).lower()] for sid in fold_info['test_ids'] if str(sid).lower() in id_to_dataset_idx]

        train_sub = torch.utils.data.Subset(full_dataset, train_indices)
        val_sub = torch.utils.data.Subset(full_dataset, val_indices)
        
        train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
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
        
        model = ClassificationPredictor(
            encoder_dict, fusion_model, model_configs["layer_size"], vocab_mod, DROPOUT, NUM_COMBINATIONS, n_classes
        ).to(DEVICE)

        if freeze_omics:
            for mod in MODALITIES_ORDER:
                for p in encoder_dict[mod].parameters(): p.requires_grad = False
        if freeze_fusion:
            for p in fusion_model.parameters(): p.requires_grad = False
            
        model.load_missing_embeds(checkpoint, DEVICE)
        
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        best_val_f1 = 0.0
        best_model_state = None
        
        for epoch in range(EPOCHS):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
            val_labels, val_preds, val_probs = evaluate(model, val_loader, DEVICE)
            val_metrics = calculate_all_metrics(val_labels, val_preds, val_probs, n_classes)
            
            torch.cuda.empty_cache() 
            print(f"Ep {epoch+1}: Train Loss {train_loss:.4f} | Val F1 {val_metrics['F1']:.4f} | ACC {val_metrics['ACC']:.4f}")
            
            if val_metrics['F1'] > best_val_f1:
                best_val_f1 = val_metrics['F1']
                best_model_state = copy.deepcopy(model.state_dict())
                print(f"  >>> New best model for this fold (F1: {best_val_f1:.4f}) <<<")

        # Load best model for this fold to perform final test (using same validation set as 'test' per original code logic)
        model.load_state_dict(best_model_state)
        test_labels, test_preds, test_probs = evaluate(model, val_loader, DEVICE)
        test_metrics = calculate_all_metrics(test_labels, test_preds, test_probs, n_classes)
        
        print(f" Fold {fold + 1} Test Metrics: ACC={test_metrics['ACC']:.4f}, F1={test_metrics['F1']:.4f}, Recall={test_metrics['Recall']:.4f}, Prec={test_metrics['Precision']:.4f}, AUPR={test_metrics['AUPR']:.4f}, AUC={test_metrics['AUC']:.4f}")
        
        test_fold_metrics.append(test_metrics)
        all_folds_labels.extend(test_labels)
        all_folds_preds.extend(test_preds)

    # --- FINAL REPORTING ---
    print(f"\n{'=' * 25} FINAL CROSS-VALIDATION SUMMARY (ON TEST SETS) {'=' * 25}")
    print("Metrics are reported as Mean ± Standard Deviation over 5 folds.\n")
    
    summary_metrics = {key: [d[key] for d in test_fold_metrics] for key in test_fold_metrics[0]}

    # 打印所有的指标 (ACC, F1, Recall, Precision, AUPR)
    for key, value in summary_metrics.items():
        mean, std = np.mean(value), np.std(value)
        print(f"{key:<15}: {mean:.4f} ± {std:.4f}")

    print("\n" + "=" * 20 + " Aggregated Classification Report (on all test folds) " + "=" * 20)
    report = classification_report(all_folds_labels, all_folds_preds, target_names=le.classes_, zero_division=0)
    print(report)

    # 计算混淆矩阵
    cm = confusion_matrix(all_folds_labels, all_folds_preds)

    # --- 新增：保存混淆矩阵为矩阵文件 ---
    # 1. 获取 LOAD_MODEL_PATH 最后一个斜杠后的内容
    model_suffix = os.path.basename(os.path.normpath(LOAD_MODEL_PATH))

    # 2. 拼接文件名
    cm_matrix_filename = f"confusion_matrix_{model_suffix}_hvg{use_hvg}.npy"
    cm_save_path = output_dir / cm_matrix_filename

    # 3. 保存为 numpy 矩阵 (.npy 格式方便后续 python 读取)
    np.save(cm_save_path, cm)
    print(f"Aggregated confusion matrix values saved to {cm_save_path}")

    # --- 原有的绘图代码 ---
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Aggregated Confusion Matrix\nAvg Test ACC: {np.mean(summary_metrics["ACC"]):.4f}') 

    # 图片文件名也可以加上这个后缀，防止覆盖
    plt_filename = f"aggregated_confusion_matrix_{model_suffix}_hvg{use_hvg}.png"
    plt.savefig(output_dir / plt_filename)
    print(f"Aggregated confusion matrix plot saved to {output_dir / plt_filename}")

    print(f"\nTraining finished. All results saved to {output_dir}")

if __name__ == '__main__':
    main()
