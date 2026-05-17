import gc
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from anndata import AnnData
from peft import LoraConfig, get_peft_model
from torch.nn import functional as F
from tqdm import tqdm

# 引入你的项目路径
sys.path.insert(0, "./") 
sys.path.append("../")

from omix.model.omni_fusion import (
    OmniFusionBlock, TrainableTextEncoder)
from omix.model.model_multi import PerformerModel
from omix.tokenizer import tokenize_and_pad_batch
from omix.tokenizer.gene_tokenizer import GeneVocab


class EmbeddingBuilder:
    def __init__(self, model_dir, device, youtu_path, model_epoch):
        self.device = device
        self.model_dir = Path(model_dir)
        self.youtu_path = youtu_path
        self.model_epoch = model_epoch
        
        # 1. 加载配置
        print(f"[Init] Loading configuration from {self.model_dir}...")
        args_path = self.model_dir / "args.json"
        if not args_path.exists():
            raise FileNotFoundError(f"Config not found at {args_path}")
            
        with open(args_path, "r") as f:
            self.config_dict = json.load(f)
            self.args = Namespace(**self.config_dict)

        # 2. 加载 Split JSON 获取 Valid IDs
        split_path = self.model_dir / "pretraining_dataset_split.json"
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found at {split_path}")
        
        with open(split_path, "r") as f:
            split_data = json.load(f)
            # 使用 set 加速查找
            self.valid_ids_set = set(split_data['valid_ids'])
            print(f"[Init] Loaded {len(self.valid_ids_set)} valid IDs from split file.")

        # 3. 加载词表
        self.vocab_dict = {}
        for mod in ['RNA', 'Protein', 'METHYL']: # 注意：根据你的预训练代码，RPPA 对应 Protein
            # 兼容 vocab 文件名可能的大小写差异
            vocab_path = self.model_dir / f"vocab_{mod.lower()}.json"
            if not vocab_path.exists() and mod == 'Protein':
                 vocab_path = self.model_dir / "vocab_protein.json" # 再次尝试

            if vocab_path.exists():
                self.vocab_dict[mod] = GeneVocab.from_file(vocab_path)
                for s in [self.args.pad_token, "<cls>", "<eoc>"]:
                    if s not in self.vocab_dict[mod]:
                        self.vocab_dict[mod].append_token(s)
            else:
                print(f"[Warning] Vocab for {mod} not found at {vocab_path}")

        self._build_models()
        self._load_weights()

    def _build_models(self):
        print("[Init] Rebuilding model architecture...")
        self.encoder_dict = {}
        
        # 重建 Biological Encoders
        for mod in ['RNA', 'Protein', 'METHYL']:
            if mod not in self.vocab_dict: continue
            
            # 读取对应的 config (layer size 可能不同)
            # 这里简化处理，通常预训练代码中会保存单独的 args.json 给每个模态
            # 如果没有单独保存，默认使用全局 config
            # 你的预训练代码逻辑是：model_configs_RNA = json.load(...)
            # 这里我们假设使用全局 args 的 layer_size，或者尝试读取子文件夹配置
            
            self.encoder_dict[mod] = PerformerModel(
                ntoken=len(self.vocab_dict[mod]),
                d_model=self.args.layer_size,
                nhead=self.args.nhead,
                d_hid=self.args.layer_size,
                nlayers=self.args.nlayers,
                vocab=self.vocab_dict[mod],
                pad_token=self.args.pad_token,
                pad_value=self.args.pad_value,
                n_input_bins=self.args.n_bins + 2
            )
            
            # Omics LoRA 逻辑 (参考预训练代码)
            if getattr(self.args, 'use_LoRA', False) and not getattr(self.args, 'text_only_lora', False):
                print(f"Applying LoRA to {mod}...")
                lora_config = LoraConfig(
                    r=self.args.lora_rank,
                    lora_alpha=self.args.lora_alpha,
                    target_modules=["to_q", "to_v"],
                    lora_dropout=self.args.dropout,
                    bias="none",
                )
                self.encoder_dict[mod] = get_peft_model(self.encoder_dict[mod], lora_config)
            
            self.encoder_dict[mod].to(self.device).eval()

        # Text Encoder
        print("   -> Building Text Encoder...")
        self.encoder_dict['TEXT'] = TrainableTextEncoder(
            model_name_or_path=self.youtu_path,
            output_dim=self.args.layer_size,
            dropout=self.args.dropout,
            trust_remote_code=True
        ).to(self.device).eval()
        
        if getattr(self.args, 'use_LoRA', False):
             text_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
             text_lora_config = LoraConfig(r=self.args.lora_rank, lora_alpha=self.args.lora_alpha, target_modules=text_target_modules, lora_dropout=self.args.dropout, bias="none", modules_to_save=["projection"])
             self.encoder_dict['TEXT'] = get_peft_model(self.encoder_dict['TEXT'], text_lora_config)

        # Fusion Model
        vocab_mod = self.args.modality_dict.copy()
        if '<pad>' not in vocab_mod: vocab_mod['<pad>'] = len(vocab_mod) # pad id
        
        self.fusion_model = OmniFusionBlock(
            self.args.num_modalities, 0, 4, 
            self.args.layer_size, self.args.num_layers_fus, 
            self.args.num_experts, self.args.num_routers, 
            self.args.top_k, self.args.nhead, self.args.dropout, 
            vocab_mod=vocab_mod,
            device=self.device
        ).to(self.device).eval()

    def _load_weights(self):
        print("[Init] Loading checkpoint weights...")
        ckpts = list(self.model_dir.glob(f"model_e{self.model_epoch}.pt"))
        if not ckpts:
             print(f"Error: No checkpoints found for epoch {self.model_epoch} in {self.model_dir}")
             return
        best_model_path = ckpts[0]
            
        print(f"   -> Loading from: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location='cpu')
        
        def remove_prefix(state_dict):
            return {k.replace("module.", ""): v for k, v in state_dict.items()}

        # === 新增：自动修复 PEFT Key 不匹配的辅助函数 ===
        def _fix_peft_keys(model, state_dict):
            model_keys = set(model.state_dict().keys())
            new_state_dict = {}
            fixed_count = 0
            
            for k, v in state_dict.items():
                # 情况 1: Key 完全匹配，直接使用
                if k in model_keys:
                    new_state_dict[k] = v
                    continue
                
                # 情况 2: 文件是 original_X (旧版/特定版), 模型需要 original_module.X
                # 例如: base_model.model.projection.original_0.weight -> base_model.model.projection.original_module.0.weight
                if "original_" in k and "original_module" not in k:
                    # 尝试将 original_0 替换为 original_module.0
                    k_fixed = k.replace("original_", "original_module.")
                    if k_fixed in model_keys:
                        new_state_dict[k_fixed] = v
                        fixed_count += 1
                        continue

                # 情况 3: 文件是 original_module.X, 模型需要 original_X
                if "original_module." in k:
                    # 尝试将 original_module.0 替换为 original_0
                    k_fixed = k.replace("original_module.", "original_")
                    if k_fixed in model_keys:
                        new_state_dict[k_fixed] = v
                        fixed_count += 1
                        continue
                
                # 如果都没匹配上，保留原样，让 load_state_dict 报错以便调试
                new_state_dict[k] = v
            
            if fixed_count > 0:
                print(f"   -> [Auto-Fix] Automatically remapped {fixed_count} PEFT keys (e.g., original_0 <-> original_module.0).")
            return new_state_dict
        # ==================================================

        # 加载 Encoders
        saved_enc_dict = checkpoint['encoder_dict']
        for mod, encoder in self.encoder_dict.items():
            if mod in saved_enc_dict:
                ckpt_sd = remove_prefix(saved_enc_dict[mod])
                
                # 应用自动修复逻辑
                ckpt_sd = _fix_peft_keys(encoder, ckpt_sd)
                
                # 加载权重
                encoder.load_state_dict(ckpt_sd)
            else:
                print(f"   -> Warning: {mod} encoder not found in checkpoint")

        # 加载 Fusion
        if 'fusion_model' in checkpoint:
            fusion_ckpt = remove_prefix(checkpoint['fusion_model'])
            self.fusion_model.load_state_dict(fusion_ckpt)
        else:
            print("   -> Warning: fusion_model not found in checkpoint")

    def process_modality(self, mod_name, h5ad_path, folder_name):
        if mod_name not in self.encoder_dict:
            return

        print(f"\n=== Processing {mod_name} from {h5ad_path} ===")
        
        # 1. Backend Loading (backed='r')
        adata = anndata.read_h5ad(h5ad_path, backed='r')
        
        # 获取该模态文件中实际存在的 IDs
        file_obs_names = adata.obs_names.to_numpy()
        
        # 2. 核心修改：使用 numpy 进行高效交集筛选
        # 逻辑：Intersection = (File IDs) ∩ (Valid IDs)
        # np.isin 返回一个布尔掩码，标记 file_obs_names 中哪些 ID 存在于 valid_ids_set 中
        print("Filtering for Valid IDs (Intersection)...")
        
        # 将 set 转为 list 传给 np.isin
        valid_mask = np.isin(file_obs_names, list(self.valid_ids_set))
        
        # 获取在文件中的行索引 (indices)
        valid_indices = np.where(valid_mask)[0]
        # 获取对应的 ID 列表 (用于保存时的索引)
        valid_ids_list = file_obs_names[valid_indices]
        
        if len(valid_indices) == 0:
            print(f"⚠️ No valid IDs found in {mod_name} file! Skipping...")
            return
            
        print(f"   -> Valid Set Size: {len(self.valid_ids_set)}")
        print(f"   -> File Size:      {len(file_obs_names)}")
        print(f"   -> Intersection:   {len(valid_indices)} samples to process.")

        # 3. 准备 Tokenizer 所需的 Gene IDs (这部分保持不变)
        vocab = self.vocab_dict[mod_name]
        pad_id = vocab[self.args.pad_token]
        
        gene_ids_list = []
        for gene in adata.var_names.tolist():
            if gene in vocab:
                gene_ids_list.append(vocab[gene])
            else:
                gene_ids_list.append(pad_id)
        gene_ids = np.array(gene_ids_list, dtype=int)
        
        if mod_name == 'RNA': max_len = self.args.max_seq_len_rna
        elif mod_name == 'Protein': max_len = self.args.max_seq_len_protein
        elif mod_name == 'METHYL': max_len = self.args.max_seq_len_methyl
        else: max_len = 512

        # 4. Batch Processing (这部分保持不变)
        z_embeddings = []
        batch_size = 8
        num_samples = len(valid_indices)
        
        with torch.no_grad():
            for i in tqdm(range(0, num_samples, batch_size), desc=f"Encoding {mod_name}"):
                # 使用筛选出的 valid_indices 来切片数据
                batch_indices = valid_indices[i : i + batch_size]
                
                # 读取数据
                layer_key = getattr(self.args, 'input_layer_key', "X_binned")
                if layer_key in adata.layers:
                    raw_batch = adata[batch_indices].layers[layer_key]
                elif "X_binned" in adata.layers:
                     raw_batch = adata[batch_indices].layers["X_binned"]
                else:
                    raw_batch = adata[batch_indices].X

                if hasattr(raw_batch, "toarray"): raw_batch = raw_batch.toarray()
                
                tokenized = tokenize_and_pad_batch(
                    raw_batch, gene_ids, max_len=max_len, vocab=vocab,
                    pad_token=self.args.pad_token, pad_value=self.args.pad_value,
                    append_cls=True, include_zero_gene=False
                )
                
                src = tokenized['genes'].to(self.device)
                values = tokenized['values'].float().to(self.device)
                src_key_padding_mask = src.eq(vocab[self.args.pad_token])

                model_impl = self.encoder_dict[mod_name]
                encoded_output, _ = model_impl(
                    src=src, values=values,
                    src_key_padding_mask=src_key_padding_mask,
                    return_seq_embedding=True
                )
                
                feat_h_seq = self.fusion_model.run_adapter(mod_name, encoded_output)
                feat_h_cls = feat_h_seq[:, 0, :]
                feat_z = self.fusion_model.simclr.project_q(feat_h_cls, mod_name)
                feat_z = F.normalize(feat_z, dim=1)
                
                z_embeddings.append(feat_z.cpu().numpy())
                del src, values, raw_batch, encoded_output

        z_embeddings = np.concatenate(z_embeddings, axis=0)
        
        # 5. 保存结果
        # 注意：这里我们只保存处理了的 valid_ids_list，而不是原始所有的 valid_ids
        print(f"Saving embeddings to {OUTPUT_DIR}...")
        
        # 从原始 adata 中复制对应的 obs 信息
        original_obs = adata.obs.iloc[valid_indices].copy()
        
        adata_emb = AnnData(X=z_embeddings, obs=original_obs)
        
        save_path = OUTPUT_DIR / f"tcga_database_{mod_name}_{folder_name}_model_e{self.model_epoch}_youtu_omics_frozen_woproject.h5ad"
        adata_emb.write(save_path)
        print(f"✅ Saved: {save_path}")
        
        if hasattr(adata, 'file'):
            adata.file.close()


# ================= 参数配置 =================
# 指向你的训练结果目录
MODEL_DIR = Path("../pretrain/save/omix_t_pretrain")
YOUTU_PATH = "./omix/Youtu_embedding"

folder_name = MODEL_DIR.name
EPOCH = 5 # 指定 Epoch

# 确保这里的路径与预训练一致
DATA_PATHS = {
    'RNA': '../../data/pretraining_data/rna_full_data.h5ad',
    'Protein': '../../data/pretraining_data/protein_all_data.h5ad', # 代码中统一用 Protein
    'METHYL': '../../data/pretraining_data/methylation_full_data.h5ad'
}

OUTPUT_DIR = Path("../../data/retrival_database_embeddings_moco_youtu")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 执行
if __name__ == "__main__":
    builder = EmbeddingBuilder(MODEL_DIR, DEVICE, YOUTU_PATH, model_epoch=EPOCH)
    
    for mod, path in DATA_PATHS.items():
        if os.path.exists(path):
            builder.process_modality(mod, path, folder_name)
        else:
            print(f"[Warning] Data file not found for {mod}: {path}")
