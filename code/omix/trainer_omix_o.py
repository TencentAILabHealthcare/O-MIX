import copy
import json
import os
import time
import traceback
import warnings
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import scanpy as sc
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from anndata import AnnData
from omix import SubsetsBatchSampler
from omix.loss import criterion_neg_log_bernoulli, masked_relative_error
from omix.tokenizer import tokenize_and_pad_batch  # 确保引用
from omix.tokenizer import random_mask_value
from omix.utils import eval_scib_metrics
from scipy.sparse import issparse
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

torch.autograd.set_detect_anomaly(True)

def create_flexmoe_input(all_ptid, modalities_data, config, rank, save_path=None):
    """
    灵活处理任意数量的模态数据，进行对齐、填充并划分为训练/验证集。

    Args:
        all_ptid (list): 所有病人的唯一ID列表 (全局基准).
        modalities_data (dict): 一个字典，包含所有模态的数据。
            格式: {
                'MODALITY_NAME_1': {
                    'tokenized': {'genes': np.array, 'values': np.array},
                    'vocab': dict,
                    'adata': AnnData object or DataFrame,
                    'char': 'R' # 用于生成组合的单字符缩写
                },
                'MODALITY_NAME_2': { ... },
                ...
            }
        config: 包含超参数的配置对象.

    Returns:
        元组 (train_data_dict, valid_data_dict, observed_idx_train, observed_idx_valid)
    """
    # 1. 建立全局 ptid -> 索引 的映射 (基石)
    id_to_idx = {id: idx for idx, id in enumerate(all_ptid)}
    n_patients = len(all_ptid)

    # 2. 初始化所有需要的数据结构
    data_dict = {}
    observed_idx_arr = np.zeros((n_patients, config.num_modalities), dtype=bool)
    modality_combinations = [''] * n_patients

    def update_modality_combinations(idx, modality_char):
        nonlocal modality_combinations
        # 保证字符不重复添加，并按字母排序
        if modality_char not in modality_combinations[idx]:
            modality_combinations[idx] = ''.join(sorted(modality_combinations[idx] + modality_char))

    # --- 核心修改：模拟 tmp 矩阵机制 ---
    text_info_dict = {}
    modality_chars_list = []
    for mod_name, mod_data in modalities_data.items():
        print(f"Processing and aligning {mod_name} modality...")
        mod_char = mod_data['char']
        modality_chars_list.append(mod_char)
        
        # === 【修改开始】 ===
        vocab = mod_data['vocab']
        adata_obj = mod_data['adata']
        
        # 1. 获取文件路径 (优先取传入的 file_path，如果没有则取 adata.filename)
        file_path = mod_data.get('file_path')
        if file_path is None and hasattr(adata_obj, 'filename'):
            file_path = adata_obj.filename

        # 2. 创建全 -1 的索引数组，-1 代表该病人缺失该模态
        indices_full = np.full((n_patients,), -1, dtype=np.int64)

        # 3. 建立映射：Patient ID -> h5ad 文件中的行号
        ptids_in_modality = adata_obj.obs.index.tolist()
        local_id_to_row_idx = {ptid: i for i, ptid in enumerate(ptids_in_modality)}

        # 4. 填充索引数组
        valid_global_indices = []
        for global_idx, ptid in enumerate(all_ptid):
            if ptid in local_id_to_row_idx:
                indices_full[global_idx] = local_id_to_row_idx[ptid]
                valid_global_indices.append(global_idx)

        if len(valid_global_indices) > 0:
            valid_global_indices = np.array(valid_global_indices)
            observed_idx_arr[valid_global_indices, config.modality_dict[mod_name]] = True
            for idx in valid_global_indices:
                update_modality_combinations(idx, mod_char)

        # 5. 只存索引和路径，不存真实数据
        data_dict[f'{mod_name}_origin_indices'] = indices_full
        data_dict[f'{mod_name}_file_path'] = file_path

    # 5. 生成模态组合索引 (现在 data_dict 中的所有数组长度都一致了)
    print("Generating modality combination indices...")

    def get_modality_combinations(modality_str):
        from itertools import combinations
        chars = [char for char in modality_str]
        all_combs = []
        for r in range(1, len(chars) + 1):
            all_combs.extend(combinations(chars, r))
        sorted_combs = [''.join(sorted(comb)) for comb in all_combs]
        full_modality_str = ''.join(sorted(chars))
        if full_modality_str in sorted_combs:
            sorted_combs.remove(full_modality_str)
            sorted_combs.insert(0, full_modality_str)
        return {comb: i for i, comb in enumerate(sorted_combs)}

    # 从输入的模态中动态构建组合字符串
    full_modality_chars = ''.join(sorted(modality_chars_list)) # e.g., 'MRP'
    combination_to_index = get_modality_combinations(full_modality_chars)
    
    # modality_combinations 已经在循环中被排好序了
    data_dict['modality_comb'] = np.array([combination_to_index.get(comb, -1) for comb in modality_combinations],
                                          dtype=np.int64)

    # 6. 划分数据集 (这部分逻辑不变，因为已经是通用的了)
    print("Splitting dataset into 90% train and 10% validation...")
    # 只对至少有一个模态的样本进行划分
    any_modality_present_indices = np.where(np.any(observed_idx_arr, axis=1))[0]
    
    if len(any_modality_present_indices) == 0:
        print("Warning: No samples with any modality data found. Returning empty dictionaries.")
        return {}, {}, np.array([]), np.array([])
        
    stratify_labels = data_dict['modality_comb'][any_modality_present_indices]
    
    unique_labels, counts = np.unique(stratify_labels, return_counts=True)

    # 2. 识别出那些只有一个样本的类别
    single_sample_labels = unique_labels[counts == 1]

    if len(single_sample_labels) > 0:
        print(f"Found {len(single_sample_labels)} classes with only one sample. These will be forced into the training set.")
        
        # 3. 创建一个布尔掩码，标记出哪些样本属于单一样本类别
        # 这个掩码的长度与 any_modality_present_indices 和 stratify_labels 相同
        is_single_sample_mask = np.isin(stratify_labels, single_sample_labels)
        
        # 4. 分离出必须放入训练集的索引，和可以安全进行划分的索引
        # 这些是全局索引
        force_train_idxs = any_modality_present_indices[is_single_sample_mask]
        
        # 这些是可以安全地进行分层划分的全局索引
        safe_to_split_indices = any_modality_present_indices[~is_single_sample_mask]
        # 以及它们对应的标签
        safe_stratify_labels = stratify_labels[~is_single_sample_mask]
        
        # 5. 仅对安全的数据进行分层划分
        train_safe_idxs, valid_idxs = np.array([], dtype=int), np.array([], dtype=int)
        if len(safe_to_split_indices) > 0:
            # 检查剩余的安全标签是否还有多样性，否则也无法分层
            if len(np.unique(safe_stratify_labels)) > 1:
                train_safe_idxs, valid_idxs = train_test_split(
                    safe_to_split_indices,
                    test_size=0.1,
                    random_state=config.seed,
                    stratify=safe_stratify_labels
                )
            else:
                # 如果剩余样本都属于同一类，则无法分层，进行普通划分
                print("Warning: Remaining samples for splitting all belong to one class. Splitting without stratification.")
                train_safe_idxs, valid_idxs = train_test_split(
                    safe_to_split_indices,
                    test_size=0.1,
                    random_state=config.seed,
                    stratify=None
                )

        # 6. 合并索引，得到最终的训练集和验证集
        train_idxs = np.concatenate((force_train_idxs, train_safe_idxs))
        # valid_idxs 已经从上面的划分中得到了，并且只包含安全的样本

    else:
        # 如果没有单一样本的类别，一切照旧
        print("No single-sample classes found. Proceeding with standard stratified split.")
        train_idxs, valid_idxs = train_test_split(
            any_modality_present_indices,
            test_size=0.1,
            random_state=config.seed,
            stratify=stratify_labels
        )

    # 7. 根据划分的索引，创建最终的 train 和 valid 数据字典
    print("Creating final train and valid data dictionaries...")
    train_data_dict = {}
    valid_data_dict = {}
    for key, value in data_dict.items():
        # 只有是数组且长度等于总病人数时，才进行切分
        if isinstance(value, (np.ndarray, list, torch.Tensor)) and len(value) == n_patients:
            train_data_dict[key] = value[train_idxs]
            valid_data_dict[key] = value[valid_idxs]
        else:
            # 字符串（如路径）或元数据直接复制
            train_data_dict[key] = value
            valid_data_dict[key] = value

    observed_idx_train = observed_idx_arr[train_idxs]
    observed_idx_valid = observed_idx_arr[valid_idxs]

    if save_path is not None:
        # 将 list 转换为 numpy array 以便使用数组索引
        all_ptid_arr = np.array(all_ptid)
        
        # 获取对应的 ID
        train_ids = all_ptid_arr[train_idxs.astype(int)].tolist()
        valid_ids = all_ptid_arr[valid_idxs.astype(int)].tolist()
        
        split_info = {
            "train_ids": train_ids,
            "valid_ids": valid_ids
        }
        
        # 保存为 json
        json_file = os.path.join(save_path, "pretraining_dataset_split.json")
        with open(json_file, "w") as f:
            json.dump(split_info, f, indent=4)
        print(f"Dataset split indices saved to {json_file}")

    return train_data_dict, valid_data_dict, observed_idx_train, observed_idx_valid


@torch.no_grad()
def momentum_update(model_q, model_k, m=0.999):
    """
    Momentum update of the key encoder
    model_k = m * model_k + (1 - m) * model_q
    """
    for param_q, param_k in zip(model_q.parameters(), model_k.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1. - m)

def collate_fn(batch):
    data, target_data, mcs, observeds = zip(*batch)
    modalities = data[0].keys()
    modalities2 = target_data[0].keys()
    collated_data = {modality: torch.tensor(np.stack([d[modality] for d in data])) for modality in modalities}
    collated_target_data = {modality: torch.tensor(np.stack([d[modality] for d in target_data])) for modality in modalities2}
    mcs = torch.tensor(mcs, dtype=torch.long)
    observeds = torch.tensor(np.vstack(observeds))
    return collated_data, collated_target_data, mcs, observeds


class MultiModalDataset(Dataset):
    def __init__(self, data_dict, observed_idx, config, vocab_dict):
        self.vocab_dict = vocab_dict
        self.config = config
        self.data_dict = data_dict
        
        # 1. 排序逻辑
        mc = np.array(data_dict['modality_comb'])
        num_modalities_per_sample = np.sum(observed_idx, axis=1)
        sorted_internal_indices = np.argsort(num_modalities_per_sample)
        
        # 2. 核心索引
        self.observed_idx = observed_idx[sorted_internal_indices]
        self.mc = mc[sorted_internal_indices]

        # --- 删除原来的 "3. TEXT" 代码块 ---
        # if 'TEXT_input_ids' in data_dict: ... (整块删除)

        # 3. Omics (原序号是4)
        self.omics_indices = {}
        self.omics_paths = {}
        self.omics_gene_ids = {} 

        for mod in config.modality_dict.keys():
            # if mod == 'TEXT': continue  <-- 这一行删掉
            self.omics_paths[mod] = data_dict[f'{mod}_file_path']
            self.omics_indices[mod] = data_dict[f'{mod}_origin_indices'][sorted_internal_indices]
            if f'{mod}_gene_ids' in data_dict:
                self.omics_gene_ids[mod] = data_dict[f'{mod}_gene_ids']
        
        self.file_handles = {}

    def _get_file_handle(self, mod, path):
        import h5py
        pid = os.getpid()
        if mod not in self.file_handles or self.file_handles[mod]['pid'] != pid:
            self.file_handles[mod] = {'f': h5py.File(path, 'r'), 'pid': pid}
        return self.file_handles[mod]['f']
    
    def __len__(self):
        return len(self.observed_idx)

    def __getitem__(self, idx):
        # 这个函数现在只返回 Raw Data，不做 Tokenize/Pad
        sample_raw = {}
        
        # 2. Omics (只读取)
        for mod in self.config.modality_dict.keys():
            
            row_idx = self.omics_indices[mod][idx]
            
            if row_idx == -1: 
                # 缺失标记：返回 None，Collate_fn 会处理
                sample_raw[mod] = None
            else:
                # 读取文件
                f = self._get_file_handle(mod, self.omics_paths[mod])
                # 兼容 layer 读取
                layer_key = getattr(self.config, 'input_layer_key', None)
                if layer_key and 'layers' in f and layer_key in f['layers']:
                    raw_data = f['layers'][layer_key][row_idx]
                elif layer_key and layer_key in f:
                    raw_data = f[layer_key][row_idx]
                else:
                    raw_data = f['X'][row_idx]
                
                # 返回原始 1D Numpy Array
                sample_raw[mod] = raw_data 

        mc = self.mc[idx]
        observed = self.observed_idx[idx]
        
        return sample_raw, mc, observed
    
class DataCollator:
    def __init__(self, config, vocab_dict, omics_gene_ids):
        """
        需要传入 vocab 和 gene_ids，以便在 collate 时使用
        """
        self.config = config
        self.vocab_dict = vocab_dict
        self.omics_gene_ids = omics_gene_ids

    def __call__(self, batch):
        from omix.tokenizer import random_mask_value, tokenize_and_pad_batch

        # Unzip batch
        # batch structure: [(sample_raw_dict, mc, observed), ...]
        raw_dicts, mcs, observeds = zip(*batch)
        
        batch_size = len(raw_dicts)
        collated_data = {}
        collated_target = {}

        # 2. Omics 处理
        for mod in self.config.modality_dict.keys():
            if mod == 'TEXT': continue
            
            # 收集当前 Batch 该模态的所有数据
            # 如果样本是 None (缺失)，我们需要构造一个全0的占位符 (为了 stack)
            # 或者更聪明的做法：记录哪些是 None，Stack 后再 Pad
            
            # 获取数据维度 (假设第一个非 None 的样本)
            n_features = len(self.omics_gene_ids[mod])
            raw_batch_list = []
            
            # 检查是否有数据，如果没有，全是缺失
            valid_sample = next((d[mod] for d in raw_dicts if d[mod] is not None), None)
            
            if valid_sample is None:
                # 整个 Batch 都没有这个模态的数据 (罕见，但可能)
                # 构造全 0 矩阵
                raw_batch_matrix = np.zeros((batch_size, n_features), dtype=np.float32)
            else:
                # 填充数据
                for d in raw_dicts:
                    if d[mod] is not None:
                        raw_batch_list.append(d[mod])
                    else:
                        # 缺失值填 0 (Tokenize 时会被忽略/Pad)
                        raw_batch_list.append(np.zeros(n_features, dtype=np.float32))
                raw_batch_matrix = np.vstack(raw_batch_list)

            # === 核心：Batch Tokenization ===
            # 这里调用一次，处理整个 Batch，这正是你想要的！
            # tokenize_and_pad_batch 会自动处理 Padding，确保 Batch 内对齐
            
            if mod == 'RNA': max_len = self.config.max_seq_len_rna
            elif mod == 'Protein': max_len = self.config.max_seq_len_protein
            elif mod == 'METHYL': max_len = self.config.max_seq_len_methyl
            # else: max_len = 512
            else: print('error')

            tokenized = tokenize_and_pad_batch(
                raw_batch_matrix,
                self.omics_gene_ids[mod],
                max_len=max_len,
                vocab=self.vocab_dict[mod],
                pad_token=self.config.pad_token,
                pad_value=self.config.pad_value,
                append_cls=True,
                include_zero_gene=False
            )
            
            genes = tokenized['genes'] # Tensor (B, max_len)
            values = tokenized['values'].float() # Tensor (B, max_len)
            
            # === Batch Masking ===
            masked_values = random_mask_value(
                values,
                mask_ratio=self.config.mask_ratio,
                mask_value=self.config.mask_value,
                pad_value=self.config.pad_value
            )
            
            collated_data[f"{mod}_genes"] = genes
            collated_data[f"{mod}_values"] = masked_values
            collated_target[f"{mod}_targets"] = values # 原始值作为 Target

        mcs = torch.tensor(mcs, dtype=torch.long)
        observeds = torch.tensor(np.vstack(observeds))
        
        return collated_data, collated_target, mcs, observeds


def uniformity_loss(x, t=2):
    """
    计算 Uniformity Loss: 衡量特征在超球面上的分布均匀程度。
    x: (N, D) 归一化后的特征向量
    t: 温度参数，默认 2
    """
    # pdist 计算两两之间的欧氏距离
    sq_dist = torch.pdist(x, p=2).pow(2)
    # 根据公式计算 loss
    return sq_dist.mul(-t).exp().mean().log()

def train(
    loader,
    encoder_dict,
    fusion_model,
    momentum_encoder_dict,
    momentum_fusion_model,
    missing_embeds_dict,
    modality_patch_counts,
    vocab_dict,
    criterion_gep_gepc,
    scaler,
    optimizer,
    scheduler,
    device,
    config,
    logger,
    epoch,
    rank,
    vocab_mod,
) -> None:
    """
    Train the model for one epoch (Pure Omics Version).
    Strategy: Decoupling
    1. Teacher: Clean Data -> MoCo Keys
    2. Student Pass A: Clean Data -> CLIP/Uniformity Queries
    3. Student Pass B: Masked Data -> MLM Input
    """
    import wandb
    modality_dict = config.modality_dict

    fusion_model.train()
    for encoder in encoder_dict.values():
        encoder.train()

    total_loss, total_gep, total_gepc, total_gate_loss, total_simsiam_loss = 0.0, 0.0, 0.0, 0.0, 0.0
    
    sorted_keys = sorted(modality_dict.keys())
    all_pairs_stats = {
        f"{m1}-{m2}": {'sum': 0.0, 'count': 0} 
        for m1, m2 in combinations(sorted_keys, 2)
    }

    log_interval = config.log_interval
    start_time = time.time()

    modality_indices = list(range(config.num_modalities))
    modality_pairs = list(combinations(modality_indices, 2))
    pad_mod_id = vocab_mod[config.pad_token] 
    num_batches = len(loader)
    sim_metrics_accumulator = {'pos': [], 'neg': []}

    for batch, (batch_data, batch_targets, batch_mcs, batch_observed) in enumerate(loader):
        if isinstance(fusion_model, DDP):
            simclr_module = fusion_model.module.simclr
        else:
            simclr_module = fusion_model.simclr
            
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        steps_to_fill_queue = simclr_module.K // (config.batch_size * world_size)
    
        if epoch == 1 and batch < steps_to_fill_queue:
            current_simsiam_weight = 0.0
            uniformity_weight = 0.0
            protein_weight = 0.0
            simclr_module.T = 0.07
        else:
            current_simsiam_weight = config.simsiam_loss_weight
            uniformity_weight = config.uniformity_weight
            protein_weight = config.protein_loss_weight
            simclr_module.T = 0.07

        batch_samples = {k: v.to(device) for k, v in batch_data.items()}
        batch_targets = {k: v.to(device) for k, v in batch_targets.items()}
        batch_mcs = batch_mcs.to(device)
        batch_observed = batch_observed.to(device)
        list_of_all_losses = []
        final_simsiam_loss_item = 0.0

        # === 0. Momentum Update ===
        with torch.no_grad():
            for key, enc_q in encoder_dict.items():
                enc_q_base = enc_q.module if isinstance(enc_q, DDP) else enc_q
                enc_k = momentum_encoder_dict[key]
                momentum_update(enc_q_base, enc_k, m=0.999)
            fusion_base = fusion_model.module if isinstance(fusion_model, DDP) else fusion_model
            momentum_update(fusion_base, momentum_fusion_model, m=0.999)

        with torch.cuda.amp.autocast(enabled=config.amp):
            
            batch_size = batch_observed.shape[0]

            # --- 容器：用于 MLM (Masked Input) ---
            fusion_input_cls_masked = []
            fusion_input_masked = []
            mod_types_list = []
            gene_token_embs_dict = {} # 用于 decoder
            modality_dict_keys = []
            
            # --- 容器：用于 CLIP/MoCo (Clean Input) ---
            pre_fusion_map_clean = {} # Student Clean Output (Query)
            pre_fusion_map_k = {}     # Teacher Clean Output (Key)

            # ============================================================
            # Phase 1: Teacher Forward (Clean Data) -> 生成 MoCo Keys
            # ============================================================
            with torch.no_grad(): 
                for modality in modality_dict.keys():
                    real_seq_len = batch_samples[f"{modality}_genes"].shape[1]
                    full_out = torch.zeros((batch_size, real_seq_len, config.layer_size), device=device)
                    mask = batch_observed[:, modality_dict[modality]]
                    
                    if mask.sum() > 0: 
                        input_genes = batch_samples[f"{modality}_genes"][mask]
                        # 【关键】Teacher 使用 Clean Data (batch_targets)
                        input_values_clean = batch_targets[f"{modality}_targets"][mask] 
                        
                        src_key_padding_mask = input_genes.eq(vocab_dict[modality][config.pad_token])
                        
                        encoded_out_k, _ = momentum_encoder_dict[modality](
                            src=input_genes, values=input_values_clean, 
                            src_key_padding_mask=src_key_padding_mask,
                            return_seq_embedding=True
                        )
                            
                        encoded_out_k = momentum_fusion_model.run_adapter(modality, encoded_out_k)
                        full_out[mask] = encoded_out_k
                    
                    pre_fusion_map_k[modality_dict[modality]] = full_out

            # ============================================================
            # Phase 2: Student Forward (Clean Data) -> 生成 CLIP Queries
            # ============================================================
            # 这一步使用 Clean Data，确保对齐任务不受 Mask 噪声影响
            for modality in modality_dict.keys():
                real_seq_len = batch_samples[f"{modality}_genes"].shape[1]
                encoded_samples_clean = torch.zeros((batch_size, real_seq_len, config.layer_size), device=device)
                mask = batch_observed[:, modality_dict[modality]]

                if mask.sum() > 0:
                    input_genes = batch_samples[f"{modality}_genes"][mask]
                    # 【关键】Student 使用 Clean Data (batch_targets)
                    input_values_clean = batch_targets[f"{modality}_targets"][mask]
                    
                    src_key_padding_mask = input_genes.eq(vocab_dict[modality][config.pad_token])
                    
                    encoded_output, _ = encoder_dict[modality].module(
                        src=input_genes,
                        values=input_values_clean,
                        src_key_padding_mask=src_key_padding_mask,
                        return_seq_embedding=True
                    )
                    encoded_samples_clean[mask] = encoded_output

                if (~mask).sum() > 0:
                    correct_missing_embeds_bank = missing_embeds_dict[modality]
                    encoded_samples_clean[~mask] = correct_missing_embeds_bank[batch_mcs[~mask], modality_dict[modality]]
                
                if isinstance(fusion_model, DDP):
                    encoded_samples_clean = fusion_model.module.run_adapter(modality, encoded_samples_clean)
                else:
                    encoded_samples_clean = fusion_model.run_adapter(modality, encoded_samples_clean)
                
                pre_fusion_map_clean[modality_dict[modality]] = encoded_samples_clean

            # ============================================================
            # Phase 3: Student Forward (Masked Data) -> 生成 MLM Input
            # ============================================================
            for modality in modality_dict.keys():
                real_seq_len = batch_samples[f"{modality}_genes"].shape[1]
                num_patches = modality_patch_counts[modality]
                modality_index = modality_dict[modality]
                
                # 初始化 Masked Pass 的画布
                encoded_samples_masked = torch.zeros((batch_size, real_seq_len, config.layer_size), device=device)
                modality_padding_mask = torch.zeros((batch_size, num_patches), dtype=torch.bool, device=device)
                mod_id = vocab_mod[modality]
                current_mod_types = torch.full((batch_size, num_patches), mod_id, dtype=torch.long, device=device)  
                
                mask = batch_observed[:, modality_index]

                if mask.sum() > 0:
                    input_genes = batch_samples[f"{modality}_genes"][mask]
                    # 【关键】Student 使用 Masked Data (batch_samples)
                    input_values_masked = batch_samples[f"{modality}_values"][mask]

                    if modality not in gene_token_embs_dict:
                        gene_embs_shape = (batch_size, real_seq_len, config.layer_size)
                        gene_token_embs_dict[modality] = torch.zeros(gene_embs_shape, device=device)

                    src_key_padding_mask = input_genes.eq(vocab_dict[modality][config.pad_token])
                    
                    encoded_output, gene_token_embs = encoder_dict[modality].module(
                        src=input_genes,
                        values=input_values_masked,
                        src_key_padding_mask=src_key_padding_mask,
                        return_seq_embedding=True
                    )
                    encoded_samples_masked[mask] = encoded_output
                    gene_token_embs_dict[modality][mask] = gene_token_embs

                if (~mask).sum() > 0:
                    correct_missing_embeds_bank = missing_embeds_dict[modality]
                    encoded_samples_masked[~mask] = correct_missing_embeds_bank[batch_mcs[~mask], modality_dict[modality]]
                
                if isinstance(fusion_model, DDP):
                    encoded_samples_masked = fusion_model.module.run_adapter(modality, encoded_samples_masked)
                else:
                    encoded_samples_masked = fusion_model.run_adapter(modality, encoded_samples_masked)
                
                current_mod_types[modality_padding_mask] = pad_mod_id

                fusion_input_cls_masked.append(encoded_samples_masked[:,0,:].unsqueeze(1).clone())
                fusion_input_masked.append(encoded_samples_masked.clone())
                mod_types_list.append(current_mod_types)
                modality_dict_keys.append(modality)

            # ==============================================================================
            # 【修复日志打印】手动计算 SimCLR 统计数据 (Monitor Only)
            # ==============================================================================
            batch_z_map = {}
            for mod_name, mod_idx in config.modality_dict.items():
                mask = batch_observed[:, mod_idx]
                # 必须确保当前 Batch 里有该模态的数据，且 pre_fusion_map_clean 中已存入
                if mask.sum() > 0 and mod_idx in pre_fusion_map_clean:
                    # 【关键修正】这里必须使用 pre_fusion_map_clean (Clean Features)
                    raw_h = pre_fusion_map_clean[mod_idx].detach() 
                    
                    # 只取 [CLS] token (index 0)
                    h_cls = raw_h[:, 0, :] 
                    
                    # 投影到 Z 空间
                    z = simclr_module.project_q(h_cls, mod_name)
                    batch_z_map[mod_name] = z
            
            # 2. 遍历所有可能的两两组合
            for m1, m2 in combinations(sorted_keys, 2):
                idx1, idx2 = modality_dict[m1], modality_dict[m2]
                
                # 如果当前 Batch 缺某个模态，跳过
                if m1 not in batch_z_map or m2 not in batch_z_map:
                    continue
                
                # 找到该对共有的样本
                mask1 = batch_observed[:, idx1]
                mask2 = batch_observed[:, idx2]
                common_mask = mask1 & mask2
                
                if common_mask.sum() > 0:
                    z1 = batch_z_map[m1][common_mask]
                    z2 = batch_z_map[m2][common_mask]
                    
                    # 计算余弦相似度
                    sim_scores = F.cosine_similarity(z1, z2, dim=1)
                    
                    # 累计到统计字典
                    key = f"{m1}-{m2}"
                    all_pairs_stats[key]['sum'] += sim_scores.sum().item()
                    all_pairs_stats[key]['count'] += sim_scores.shape[0]

            
            # ================= Phase 4: Loss Calculation =================

            # --- 4.1 Uniformity Loss (使用 Clean Features) ---
            uni_loss_sum = torch.tensor(0.0, device=device)
            uni_count = 0
            for mod_name, mod_idx in config.modality_dict.items():
                mask = batch_observed[:, mod_idx]
                if mask.sum() > 1:
                    # 使用 Phase 2 的 Clean 结果
                    raw_h = pre_fusion_map_clean[mod_idx] 
                    valid_h = raw_h[mask]
                    valid_h_cls = valid_h[:, 0, :]
                    
                    simclr_module = fusion_model.module.simclr if isinstance(fusion_model, DDP) else fusion_model.simclr
                    z = simclr_module.project_q(valid_h_cls, mod_name)
                    z = F.normalize(z, dim=1)
                    current_uni_loss = uniformity_loss(z)
                    if mod_name == 'Protein':
                        current_uni_loss = current_uni_loss * protein_weight 
                    
                    uni_loss_sum += current_uni_loss
                    uni_count += 1
            final_uni_loss = torch.tensor(0.0, device=device)
            if uni_count > 0: final_uni_loss = uni_loss_sum / uni_count
            list_of_all_losses.append(uniformity_weight * final_uni_loss)
            if rank == 0 and batch % log_interval == 0:
                wandb.log({"train/uniformity_loss": final_uni_loss.item()}, commit=False)

            # --- 4.2 Fusion Forward & SimCLR (使用 Clean Maps) ---
            mod_types = torch.cat(mod_types_list, dim=1)
            
            # 【核心】：SimCLR 使用 Clean Map，Cross-Attention 输入使用 Masked Map
            hidden_x, simclr_loss_val, batch_sim_stats = fusion_model(
                    *fusion_input_cls_masked, 
                    expert_indices=batch_mcs, 
                    mod_types=mod_types,
                    calculate_simclr=True,
                    batch_observed=batch_observed,
                    pre_fusion_map=pre_fusion_map_clean,  # <--- Clean Query
                    pre_fusion_map_k=pre_fusion_map_k,    # <--- Clean Key
                    modality_pairs=modality_pairs,
                    text_loss_weight=config.text_loss_weight,
                    bio_loss_weight=config.bio_loss_weight,
                    protein_loss_weight=protein_weight,
                    training=True
                )
            
            if batch_sim_stats:
                sim_metrics_accumulator['pos'].append(batch_sim_stats['avg_pos_sim'])
                sim_metrics_accumulator['neg'].append(batch_sim_stats['avg_neg_sim'])
            
            list_of_all_losses.append(current_simsiam_weight * simclr_loss_val)
            total_simsiam_loss += simclr_loss_val.item()

            # --- 4.3 MLM Loss (使用 Masked Input 的输出) ---
            tmp_gepc_loss = 0.0
            tmp_gep_loss = 0.0

            for index, modality in enumerate(modality_dict_keys):
                present_mask = batch_observed[:, config.modality_dict[modality]]
                if not present_mask.any():
                    # Dummy Loss
                    current_encoder_module = encoder_dict[modality].module if isinstance(encoder_dict[modality], DDP) else encoder_dict[modality]
                    dummy_loss_all = sum(p.sum() for p in current_encoder_module.parameters()) * 0.0
                    list_of_all_losses.append(dummy_loss_all)
                else:
                    # 获取 Phase 3 (Masked) 的 Fusion Input
                    original_sequence_subset = fusion_input_masked[index][present_mask]
                    condition_hidden_x_subset = torch.squeeze(torch.cat(hidden_x, dim=-1)[present_mask], dim=1)
                    
                    # AdaLN
                    modality_hidden_x_subset = fusion_model.module.get_adaln(original_sequence_subset, condition_hidden_x_subset)
                    
                    modality_gene_embs_subset = gene_token_embs_dict[modality][present_mask]
                    
                    # Target 是 Clean 的
                    targets_subset = batch_targets[f"{modality}_targets"][present_mask]
                    # Mask 掩码根据 Masked Values 计算
                    values_subset = batch_samples[f"{modality}_values"][present_mask]
                    masked_positions = values_subset.eq(config.mask_value)

                    output_dict = encoder_dict[modality].module.decode_only(modality_hidden_x_subset, modality_gene_embs_subset)

                    loss_gep = criterion_gep_gepc(
                        output_dict["mlm_output"], targets_subset, masked_positions
                    )
                    loss_gepc = criterion_gep_gepc(
                        output_dict["mvc_output"], targets_subset, masked_positions
                    )

                    list_of_all_losses.append(loss_gep)
                    list_of_all_losses.append(loss_gepc)

                    tmp_gep_loss += loss_gep.item()
                    tmp_gepc_loss += loss_gepc.item()

            gate_loss = fusion_model.module.gate_loss()
            list_of_all_losses.append(config.gate_loss_weight * gate_loss)
            
            loss = sum(list_of_all_losses)
        
        # Backward & Logging
        original_loss_item = loss.item()
        loss = loss / config.gradient_accumulation_steps
        scaler.scale(loss).backward()
        
        if (batch + 1) % config.gradient_accumulation_steps == 0 or (batch + 1) == num_batches:
            scaler.unscale_(optimizer)
            all_params = []
            for param_group in optimizer.param_groups:
                all_params.extend(param_group['params'])
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                torch.nn.utils.clip_grad_norm_(all_params, 1.0, error_if_nonfinite=False if scaler.is_enabled() else True)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += original_loss_item
        total_gep += tmp_gep_loss
        total_gepc += tmp_gepc_loss
        total_gate_loss += gate_loss.item()
        total_simsiam_loss += final_simsiam_loss_item
        
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_gep = total_gep / log_interval if config.GEP else 0.0
            cur_gepc = total_gepc / log_interval if config.GEPC else 0.0
            cur_gate_loss = total_gate_loss / log_interval
            cur_simsiam_loss = total_simsiam_loss / log_interval
            sim_log_str = ""
            sim_posneg_str = ""
            
            if rank == 0:
                temp_str = ""
                if batch_sim_stats and 'temp' in batch_sim_stats:
                    temp_str = f" | T: {batch_sim_stats['temp']:.4f}"
                if sim_metrics_accumulator['pos']: 
                    avg_pos = sum(sim_metrics_accumulator['pos']) / len(sim_metrics_accumulator['pos'])
                    avg_neg = sum(sim_metrics_accumulator['neg']) / len(sim_metrics_accumulator['neg'])
                    gap = avg_pos - avg_neg
                    sim_posneg_str = f"Pos Sim: {avg_pos:.4f} | Neg Sim: {avg_neg:.4f} | GAP: {gap:.4f}{temp_str}"

                for pair_key, stats in all_pairs_stats.items():
                    if stats['count'] > 0:
                        avg_sim = stats['sum'] / stats['count']
                        sim_log_str += f" | {pair_key}: {avg_sim:.4f}"
                        wandb.log({f"train/sim_{pair_key}": avg_sim}, commit=False)
                    else:
                        sim_log_str += f" | {pair_key}: ----"
                
                logger.info(
                    f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                    f"lr {lr:05.5f} | ms/batch {ms_per_batch:5.2f} | "
                    f"loss {cur_loss:5.2f} | "
                    f"gate {cur_gate_loss:5.2f} |"
                    f"simclr {cur_simsiam_loss:5.2f} |"
                    f"{sim_log_str} |"
                    f"{sim_posneg_str} | " +
                    f"uniform :{final_uni_loss.item():5.2f} | "
                    + (f"gep {cur_gep:5.2f} | " if config.GEP else "")
                    + (f"gepc {cur_gepc:5.2f} | " if config.GEPC else "")
                )
            
            # 重置
            total_loss = 0; total_gep = 0; total_gepc = 0; total_gate_loss = 0; total_simsiam_loss = 0
            start_time = time.time()
            sim_metrics_accumulator = {'pos': [], 'neg': []}
            all_pairs_stats = {f"{m1}-{m2}": {'sum': 0.0, 'count': 0} for m1, m2 in combinations(sorted_keys, 2)}

        del loss, hidden_x, batch_samples, batch_targets, batch_mcs, batch_observed, gate_loss


def define_wandb_metrcis():
    import wandb

    wandb.define_metric("valid/loss", summary="min", step_metric="epoch")
    wandb.define_metric("test/avg_bio", summary="max")


import time
import warnings
from itertools import combinations

import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

# 假设 train 函数中使用的其他依赖项 (如 time, warnings) 已经导入

def evaluate(
    loader,
    encoder_dict,
    fusion_model,
    missing_embeds_dict,
    modality_patch_counts,
    vocab_dict,
    criterion_gep_gepc,
    device,
    config,
    epoch,
    logger,
    rank,
    vocab_mod
) -> float:
    """
    Evaluate the model on the validation data (Pure Omics Version).
    Strictly decoupled: 
    - Pass 1: Clean Data -> CLIP metrics (Uniformity, Sim)
    - Pass 2: Masked Data -> MLM metrics (Reconstruction)
    """
    import wandb
    modality_dict = config.modality_dict

    # 1. 切换到评估模式
    fusion_model.eval()
    for encoder in encoder_dict.values():
        encoder.eval()

    # 在评估时使用原始模型
    unwrapped_fusion_model = fusion_model.module if isinstance(fusion_model, DDP) else fusion_model
    unwrapped_encoder_dict = {
        name: (model.module if isinstance(model, DDP) else model) 
        for name, model in encoder_dict.items()
    }

    total_loss, total_gep, total_gepc, total_gate_loss, total_simsiam_loss = 0.0, 0.0, 0.0, 0.0, 0.0
    
    sorted_keys = sorted(modality_dict.keys())
    all_pairs_stats = {
        f"{m1}-{m2}": {'sum': 0.0, 'count': 0} 
        for m1, m2 in combinations(sorted_keys, 2)
    }
    sim_metrics_accumulator = {'pos': [], 'neg': []}
    
    modality_indices = list(range(config.num_modalities))
    modality_pairs = list(combinations(modality_indices, 2))
    pad_mod_id = vocab_mod[config.pad_token]

    # 初始化 avg_uni_loss 防止除零错误
    avg_uni_loss = 0.0

    # 2. 禁用梯度计算
    with torch.no_grad():
        for batch, (batch_data, batch_targets, batch_mcs, batch_observed) in enumerate(tqdm(loader, disable=(rank != 0))):
            batch_samples = {k: v.to(device) for k, v in batch_data.items()}
            batch_targets = {k: v.to(device) for k, v in batch_targets.items()}
            batch_mcs = batch_mcs.to(device)
            batch_observed = batch_observed.to(device)
            list_of_all_losses = []
            
            batch_size = batch_observed.shape[0]

            with torch.cuda.amp.autocast(enabled=config.amp):
                
                # =========================================================
                # Pass 1: Clean Data (用于 CLIP / Uniformity)
                # =========================================================
                pre_fusion_map_clean = {} 
                
                for modality in modality_dict.keys():
                    real_seq_len = batch_samples[f"{modality}_genes"].shape[1]
                    mask = batch_observed[:, modality_dict[modality]]
                    
                    # 准备 Clean Output 容器
                    full_out_clean = torch.zeros((batch_size, real_seq_len, config.layer_size), device=device)

                    if mask.sum() > 0:
                        input_genes = batch_samples[f"{modality}_genes"][mask]
                        # 【关键】使用 Clean Targets (Clean Values)
                        input_values_clean = batch_targets[f"{modality}_targets"][mask] 
                        
                        src_key_padding_mask = input_genes.eq(vocab_dict[modality][config.pad_token])
                        
                        encoded_out_clean, _ = unwrapped_encoder_dict[modality](
                            src=input_genes, values=input_values_clean, 
                            src_key_padding_mask=src_key_padding_mask,
                            return_seq_embedding=True
                        )
                        full_out_clean[mask] = encoded_out_clean

                    # 补全缺失模态 (为了 Adapter 对齐)
                    if (~mask).sum() > 0:
                        correct_missing_embeds_bank = missing_embeds_dict[modality]
                        full_out_clean[~mask] = correct_missing_embeds_bank[batch_mcs[~mask], modality_dict[modality]]

                    full_out_clean = unwrapped_fusion_model.run_adapter(modality, full_out_clean)
                    pre_fusion_map_clean[modality_dict[modality]] = full_out_clean

                # =========================================================
                # Pass 2: Masked Data (用于 MLM Reconstruction)
                # =========================================================
                fusion_input_cls_masked = []
                fusion_input_masked = []
                mod_types_list = []
                modality_dict_keys = []
                gene_token_embs_dict = {}

                for modality in modality_dict.keys():
                    real_seq_len = batch_samples[f"{modality}_genes"].shape[1]
                    num_patches = modality_patch_counts[modality]
                    mod_id = vocab_mod[modality]
                    
                    # 初始化容器
                    encoded_samples_masked = torch.zeros((batch_size, real_seq_len, config.layer_size), device=device)
                    current_mod_types = torch.full((batch_size, num_patches), mod_id, dtype=torch.long, device=device)
                    modality_padding_mask = torch.zeros((batch_size, num_patches), dtype=torch.bool, device=device)
                    
                    mask = batch_observed[:, modality_dict[modality]]

                    if mask.sum() > 0:
                        input_genes = batch_samples[f"{modality}_genes"][mask]
                        # 【关键】使用 Masked Samples (Masked Values)
                        input_values_masked = batch_samples[f"{modality}_values"][mask]
                        
                        if modality not in gene_token_embs_dict:
                            gene_embs_shape = (batch_size, real_seq_len, config.layer_size)
                            gene_token_embs_dict[modality] = torch.zeros(gene_embs_shape, device=device)

                        src_key_padding_mask = input_genes.eq(vocab_dict[modality][config.pad_token])
                        
                        encoded_output, gene_token_embs = unwrapped_encoder_dict[modality](
                            src=input_genes, values=input_values_masked,
                            src_key_padding_mask=src_key_padding_mask,
                            return_seq_embedding=True
                        )
                        encoded_samples_masked[mask] = encoded_output
                        gene_token_embs_dict[modality][mask] = gene_token_embs

                    # 补全缺失
                    if (~mask).sum() > 0:
                        correct_missing_embeds_bank = missing_embeds_dict[modality]
                        encoded_samples_masked[~mask] = correct_missing_embeds_bank[batch_mcs[~mask], modality_dict[modality]]
                    
                    encoded_samples_masked = unwrapped_fusion_model.run_adapter(modality, encoded_samples_masked)

                    current_mod_types[modality_padding_mask] = pad_mod_id
                    fusion_input_cls_masked.append(encoded_samples_masked[:,0,:].unsqueeze(1).clone())
                    fusion_input_masked.append(encoded_samples_masked.clone())
                    mod_types_list.append(current_mod_types)
                    modality_dict_keys.append(modality)

                # =========================================================
                # Metrics 1: Uniformity (Based on Clean)
                # =========================================================
                uni_loss_sum = 0.0
                uni_count = 0
                for mod_name, mod_idx in config.modality_dict.items():
                    mask = batch_observed[:, mod_idx]
                    if mask.sum() > 1:
                        # 使用 Pass 1 的 Clean 结果
                        raw_h = pre_fusion_map_clean[mod_idx].detach() 
                        h_cls = raw_h[mask][:, 0, :]
                        z = unwrapped_fusion_model.get_z(h_cls, mod_name)
                        sq_dist = torch.pdist(z, p=2).pow(2)
                        uni_l = sq_dist.mul(-2).exp().mean().log()
                        uni_loss_sum += uni_l.item()
                        uni_count += 1
                if uni_count > 0:
                    avg_uni_loss = uni_loss_sum / uni_count

                # =========================================================
                # Metrics 2: Similarity Statistics (Based on Clean)
                # =========================================================
                batch_z_map = {}
                for mod_name, mod_idx in modality_dict.items():
                    mask = batch_observed[:, mod_idx]
                    if mask.sum() > 0:
                        raw_h = pre_fusion_map_clean[mod_idx].detach()
                        h_cls = raw_h[:, 0, :] 
                        z = unwrapped_fusion_model.get_z(h_cls, mod_name)
                        batch_z_map[mod_name] = z

                for m1, m2 in combinations(sorted_keys, 2):
                    if m1 not in batch_z_map or m2 not in batch_z_map: continue
                    mod_idx1, mod_idx2 = modality_dict[m1], modality_dict[m2]
                    mask1 = batch_observed[:, mod_idx1]
                    mask2 = batch_observed[:, mod_idx2]
                    common_mask = mask1 & mask2
                    
                    if common_mask.sum() > 0:
                        z1 = batch_z_map[m1][common_mask]
                        z2 = batch_z_map[m2][common_mask]
                        sim_scores = F.cosine_similarity(z1, z2, dim=1)
                        
                        key = f"{m1}-{m2}"
                        all_pairs_stats[key]['sum'] += sim_scores.sum().item()
                        all_pairs_stats[key]['count'] += sim_scores.shape[0]

                # =========================================================
                # Metrics 3: OmniFusionBlock Forward & SimCLR Loss
                # =========================================================
                mod_types = torch.cat(mod_types_list, dim=1)
                
                # 在 Valid 中，我们通常计算 In-Batch Similarity
                # 将 Clean Map 同时作为 Query 和 Key 传入
                hidden_x, simclr_loss_val, batch_sim_stats = unwrapped_fusion_model(
                    *fusion_input_cls_masked, 
                    expert_indices=batch_mcs, 
                    mod_types=mod_types,
                    calculate_simclr=True,
                    batch_observed=batch_observed,
                    pre_fusion_map=pre_fusion_map_clean,    # Query (Clean)
                    pre_fusion_map_k=pre_fusion_map_clean,  # Key (Clean - Self Contrast)
                    modality_pairs=modality_pairs,
                    text_loss_weight=config.text_loss_weight,
                    bio_loss_weight=config.bio_loss_weight,
                    training=False # 不更新 Queue
                )
                
                list_of_all_losses.append(config.simsiam_loss_weight * simclr_loss_val)
                total_simsiam_loss += simclr_loss_val.item()
                if batch_sim_stats:
                    sim_metrics_accumulator['pos'].append(batch_sim_stats['avg_pos_sim'])
                    sim_metrics_accumulator['neg'].append(batch_sim_stats['avg_neg_sim'])
                
                # --- MLM Loss (Masked Input -> Clean Target) ---
                tmp_gep_loss = 0.0
                tmp_gepc_loss = 0.0
                for index, modality in enumerate(modality_dict_keys):
                    present_mask = batch_observed[:, config.modality_dict[modality]]
                    if present_mask.any():
                        # 使用 Masked Pass 的 Hidden States
                        original_sequence_subset = fusion_input_masked[index][present_mask]
                        condition_hidden_x_subset = torch.squeeze(torch.cat(hidden_x, dim=-1)[present_mask], dim=1)
                        modality_hidden_x_subset = unwrapped_fusion_model.get_adaln(original_sequence_subset, condition_hidden_x_subset)
                        
                        modality_gene_embs_subset = gene_token_embs_dict[modality][present_mask]
                        
                        # 输入: Masked Values 的 mask 位置
                        values_subset = batch_samples[f"{modality}_values"][present_mask]
                        # 目标: Clean Values
                        targets_subset = batch_targets[f"{modality}_targets"][present_mask]
                        
                        output_dict = unwrapped_encoder_dict[modality].decode_only(modality_hidden_x_subset, modality_gene_embs_subset)
                        masked_positions = values_subset.eq(config.mask_value)

                        if masked_positions.any():
                            loss_gep = criterion_gep_gepc(output_dict["mlm_output"], targets_subset, masked_positions)
                            loss_gepc = criterion_gep_gepc(output_dict["mvc_output"], targets_subset, masked_positions)
                            list_of_all_losses.append(loss_gep)
                            list_of_all_losses.append(loss_gepc)
                            tmp_gep_loss += loss_gep.item()
                            tmp_gepc_loss += loss_gepc.item()

                total_gep += tmp_gep_loss
                total_gepc += tmp_gepc_loss

                gate_loss = unwrapped_fusion_model.gate_loss()
                list_of_all_losses.append(config.gate_loss_weight * gate_loss)
                total_gate_loss += gate_loss.item()
                
                if list_of_all_losses:
                    final_loss_per_batch = torch.stack(list_of_all_losses).sum()
                    total_loss += final_loss_per_batch.item()

    # --- 日志记录和返回值 ---
    num_batches = len(loader)
    if num_batches == 0:
        if rank == 0: logger.warning("Validation loader is empty. Returning 0 loss.")
        return 0.0

    avg_loss = total_loss / num_batches
    avg_gep = total_gep / num_batches
    avg_gepc = total_gepc / num_batches
    avg_gate = total_gate_loss / num_batches
    avg_simsiam = total_simsiam_loss / num_batches

    sim_log_dict = {}
    sim_log_str = ""
    for pair_key, stats in all_pairs_stats.items():
        if stats['count'] > 0:
            avg_sim = stats['sum'] / stats['count']
            sim_log_dict[f"valid/sim_{pair_key}"] = avg_sim
            sim_log_str += f" | {pair_key}: {avg_sim:.4f}"
        else:
            sim_log_str += f" | {pair_key}: ----"

    sim_posneg_str = ""
    if sim_metrics_accumulator['pos']:
        avg_pos = sum(sim_metrics_accumulator['pos']) / len(sim_metrics_accumulator['pos'])
        avg_neg = sum(sim_metrics_accumulator['neg']) / len(sim_metrics_accumulator['neg'])
        gap = avg_pos - avg_neg
        sim_posneg_str = f"Pos Sim: {avg_pos:.4f} | Neg Sim: {avg_neg:.4f} | GAP: {gap:.4f}"
    
    if rank == 0:
        logger.info(
            f"| epoch {epoch:3d} | valid loss {avg_loss:5.2f} | "
            f"gate {avg_gate:5.2f} | "
            f"simclr {avg_simsiam:5.2f} |"
            f"{sim_log_str} | "
            f"{sim_posneg_str} | "
            + f"uni: {avg_uni_loss:5.2f} |"
            + (f"gep {avg_gep:5.2f} |" if config.GEP else "")
            + (f"gepc {avg_gepc:5.2f} |" if config.GEPC else "")
        )

        wandb_dict = {
            "valid/loss": avg_loss,
            "valid/gep_loss": avg_gep,
            "valid/gepc_loss": avg_gepc,
            "valid/gate_loss": avg_gate,
            "valid/simclr_loss": avg_simsiam,
            "epoch": epoch,
        }
        if sim_metrics_accumulator['pos']:
            wandb_dict["valid/sim_pos"] = avg_pos
            wandb_dict["valid/sim_neg"] = avg_neg
            wandb_dict["valid/sim_gap"] = gap

        wandb_dict.update(sim_log_dict)
        wandb.log(wandb_dict)
        
    return avg_loss


def predict(
    model: nn.Module,
    loader: DataLoader,
    vocab,
    config,
    device,
) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()

    predictions = []
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            celltype_labels = batch_data["celltype_labels"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[config.pad_token])

            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels
                    if config.use_batch_labels or config.DSBN
                    else None,
                    CLS=config.CLS,
                    MVC=config.GEPC,
                    ECS=config.ESC,
                )

                output_values = output_dict["cls_output"]
                preds = output_values.argmax(1).cpu().numpy()
                predictions.append(preds)

    return np.concatenate(predictions, axis=0)


# %% inference
def test(
    model: nn.Module, adata: DataLoader, gene_ids, vocab, config, device, logger
) -> float:
    all_counts = (
        adata.layers[config.input_layer_key].toarray()
        if issparse(adata.layers[config.input_layer_key])
        else adata.layers[config.input_layer_key]
    )

    celltypes_labels = adata.obs["celltype_id"].tolist()  # make sure count from 0
    celltypes_labels = np.array(celltypes_labels)

    batch_ids = adata.obs["batch_id"].tolist()
    batch_ids = np.array(batch_ids)

    tokenized_test = tokenize_and_pad_batch(
        all_counts,
        gene_ids,
        max_len=config.max_seq_len,
        vocab=vocab,
        pad_token=config.pad_token,
        pad_value=config.pad_value,
        append_cls=True,  # append <cls> token at the beginning
        include_zero_gene=config.include_zero_gene,
    )

    input_values_test = random_mask_value(
        tokenized_test["values"],
        mask_ratio=config.mask_ratio,
        mask_value=config.mask_value,
        pad_value=config.pad_value,
    )

    test_data_pt = {
        "gene_ids": tokenized_test["genes"],
        "values": input_values_test,
        "target_values": tokenized_test["values"],
        "batch_labels": torch.from_numpy(batch_ids).long(),
        "celltype_labels": torch.from_numpy(celltypes_labels).long(),
    }

    test_loader = DataLoader(
        dataset=SeqDataset(test_data_pt),
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=min(len(os.sched_getaffinity(0)), config.batch_size // 2),
        pin_memory=True,
    )

    model.eval()
    predictions = predict(
        model,
        test_loader,
        vocab,
        config,
        device,
    )

    # compute accuracy, precision, recall, f1
    accuracy = accuracy_score(celltypes_labels, predictions)
    precision = precision_score(celltypes_labels, predictions, average="macro")
    recall = recall_score(celltypes_labels, predictions, average="macro")
    macro_f1 = f1_score(celltypes_labels, predictions, average="macro")
    micro_f1 = f1_score(celltypes_labels, predictions, average="micro")

    logger.info(
        f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, "
        f"Macro F1: {macro_f1:.3f}, Micro F1: {micro_f1:.3f}"
    )

    results = {
        "test/accuracy": accuracy,
        "test/precision": precision,
        "test/recall": recall,
        "test/macro_f1": macro_f1,
        "test/micro_f1": micro_f1,
    }

    return predictions, celltypes_labels, results


def eval_testdata(
    model: nn.Module,
    adata_t: AnnData,
    gene_ids,
    vocab,
    config,
    logger,
    include_types: List[str] = ["cls"],
) -> Optional[Dict]:
    """evaluate the model on test dataset of adata_t"""
    model.eval()

    # copy adata_t to avoid reuse previously computed results stored in adata_t
    adata_t = adata_t.copy()

    all_counts = (
        adata_t.layers[config.input_layer_key].toarray()
        if issparse(adata_t.layers[config.input_layer_key])
        else adata_t.layers[config.input_layer_key]
    )

    celltypes_labels = adata_t.obs["celltype"].tolist()
    celltypes_labels = np.array(celltypes_labels)

    batch_ids = adata_t.obs["batch_id"].tolist()
    batch_ids = np.array(batch_ids)

    # Evaluate cls cell embeddings
    if "cls" in include_types:
        logger.info("Evaluating cls cell embeddings")
        tokenized_all = tokenize_and_pad_batch(
            all_counts,
            gene_ids,
            max_len=config.max_seq_len,
            vocab=vocab,
            pad_token=config.pad_token,
            pad_value=config.pad_value,
            append_cls=True,  # append <cls> token at the beginning
            include_zero_gene=config.include_zero_gene,
        )
        all_gene_ids, all_values = tokenized_all["genes"], tokenized_all["values"]
        src_key_padding_mask = all_gene_ids.eq(vocab[config.pad_token])
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=config.amp):
            cell_embeddings = model.encode_batch(
                all_gene_ids,
                all_values.float(),
                src_key_padding_mask=src_key_padding_mask,
                batch_size=config.batch_size,
                batch_labels=torch.from_numpy(batch_ids).long()
                if config.DSBN or config.DAR or config.use_batch_labels
                else None,
                time_step=0,
                return_np=True,
            )
        cell_embeddings = cell_embeddings / np.linalg.norm(
            cell_embeddings, axis=1, keepdims=True
        )

        adata_t.obsm["X_omix"] = cell_embeddings

        results = {}
        try:
            results = eval_scib_metrics(adata_t)
        except Exception as e:
            traceback.print_exc()
            logger.error(e)

        sc.pp.neighbors(adata_t, use_rep="X_omix")
        sc.tl.umap(adata_t, min_dist=0.3)
        fig = sc.pl.umap(
            adata_t,
            color=["str_batch"],
            title=[f"batch, avg_bio = {results.get('avg_bio', 0.0):.4f}"],
            frameon=False,
            return_fig=True,
            show=False,
        )

        results["batch_umap"] = fig

        sc.pp.neighbors(adata_t, use_rep="X_omix")
        sc.tl.umap(adata_t, min_dist=0.3)
        fig = sc.pl.umap(
            adata_t,
            color=["celltype"],
            title=[
                f"celltype, avg_bio = {results.get('avg_bio', 0.0):.4f}",
            ],
            frameon=False,
            return_fig=True,
            show=False,
        )

        results["celltype_umap"] = fig

    if len(include_types) == 1:
        return results
        return results
