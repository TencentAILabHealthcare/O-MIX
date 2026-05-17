import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from anndata import AnnData
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

sys.path.insert(0, "./") 
sys.path.append("../")
from peft import LoraConfig, get_peft_model
from omix.model.omni_fusion import (
    OmniFusionBlock, TrainableTextEncoder)

# ================= 配置区域 =================
MODEL_DIR = Path("../pretrain/save/omix_t_pretrain")
SPLIT_JSON_PATH = MODEL_DIR / "pretraining_dataset_split.json"
FOLDER_NAME = MODEL_DIR.name # 获取文件夹名用于拼接文件名
EPOCH = 5

# 文本原始数据 (JSON)
TEXT_JSON_PATH = '../../data/pretraining_data/generated_files/textual_annotations_llama_clean.json'
YOUTU_PATH = "./omix/Youtu_embedding"

# Bio Database 路径 (Step 1 生成的 Bio 缓存，也是 Text 缓存保存的位置)
DATABASE_DIR = "../../data/retrival_database_embeddings_moco_youtu"

# 评估设置
SOURCE_MODALITY = 'TEXT' 
TARGET_MODALITY = 'METHYL' # 可选: RNA, Protein, METHYL, TEXT

SAMPLE_SIZE = 100 # 设置为 None 以使用全部 Valid 集
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
KMEANS_RANDOM_SEED = 42
KMEANS_N_INIT = 10

SELECT_STRATEGY = "Kmeans" # Random, Kmeans

# ===========================================

class RetrievalEvaluator:
    def __init__(self, model_dir, device, youtu_path, model_epoch, folder_name, database_dir):
        self.device = device
        self.model_dir = Path(model_dir)
        self.youtu_path = youtu_path
        self.model_epoch = model_epoch
        self.folder_name = folder_name
        self.database_dir = Path(database_dir)
        self.database_dir.mkdir(exist_ok=True, parents=True)
        
        self._load_config_and_model()

    def _load_text_dict(self):
        with open(TEXT_JSON_PATH, 'r') as f:
            text_dict = json.load(f)
        return text_dict

    def _select_kmeans_anchors(self, ids, src_ids, src_data, n_clusters, seed=42, n_init=10,
                           aligned_text_list=None):
        """
        在 SOURCE_MODALITY 对应的 embedding 空间做 KMeans，
        选取每个簇中心最近的样本作为 anchor。
        
        参数:
            ids: 已经对齐好的 final_ids
            src_ids: source 全量 id 列表
            src_data: source 对应的 AnnData(backed) 或 AnnData
            n_clusters: 需要选多少个 anchor
            aligned_text_list: 与 ids 一一对齐的文本列表；如果当前模态对中包含 TEXT，则传入，否则为 None
        返回:
            selected_ids: KMeans 选出的 anchor IDs
        """
        if len(ids) < n_clusters:
            raise ValueError(
                f"Number of aligned samples ({len(ids)}) is smaller than n_clusters ({n_clusters})."
            )

        print(f"   -> Running KMeans on {len(ids)} aligned source samples (n_clusters={n_clusters})...")

        idx_map = {pid: i for i, pid in enumerate(src_ids)}
        indices = [idx_map[pid] for pid in ids]

        matrix = src_data[indices].X
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()

        matrix = np.asarray(matrix, dtype=np.float32)

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=seed,
            n_init=n_init
        )
        kmeans.fit(matrix)

        closest_idx_in_pool, _ = pairwise_distances_argmin_min(
            kmeans.cluster_centers_,
            matrix
        )

        selected_ids = ids[closest_idx_in_pool]

        print(f"\n[Analysis] Extracting details for the {len(closest_idx_in_pool)} selected prototypes...")

        prototype_data = []
        for cluster_idx, pool_idx in enumerate(closest_idx_in_pool):
            s_id = ids[pool_idx]

            row = {
                "Cluster_ID": cluster_idx,
                "Sample_Index": int(pool_idx),
                "Sample_ID": s_id,
            }

            if aligned_text_list is not None:
                row["Text_Content"] = aligned_text_list[pool_idx]

            prototype_data.append(row)

        df_prototypes = pd.DataFrame(prototype_data)

        print("\n--- Prototype Preview (Top 5) ---")
        preview_df = df_prototypes.copy()
        if "Text_Content" in preview_df.columns:
            preview_df["Text_Content"] = preview_df["Text_Content"].apply(
                lambda x: x[:50] + "..." if isinstance(x, str) and len(x) > 50 else x
            )
            print(preview_df[["Cluster_ID", "Sample_ID", "Text_Content"]].head(5))
        else:
            print(preview_df[["Cluster_ID", "Sample_ID"]].head(5))

        save_path = self.database_dir / "kmeans_prototypes_100.csv"
        df_prototypes.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ Full prototype list saved to: {save_path}")

        print(f"   -> KMeans selected {len(selected_ids)} anchor samples.")
        print(f"   -> Example anchor IDs: {selected_ids[:5].tolist() if len(selected_ids) >= 5 else selected_ids.tolist()}")

        return selected_ids
    
    def _load_state_dict_with_debug(self, model, state_dict, module_name="Model"):
        missing_keys, unexpected_keys = model.load_state_dict(state_dict)
        
        print(f"\n[Weight Load Debug] {module_name}:")
        if len(missing_keys) > 0:
            # 过滤掉 LoRA 相关的 missing keys
            real_missing = [k for k in missing_keys if 'lora' not in k]
            if len(real_missing) > 0:
                print(f"  ‼️ CRITICAL MISSING (Non-LoRA): {real_missing[:5]}")
            else:
                print(f"  ✅ Missing keys seem to be LoRA related (Safe).")
        
        if len(missing_keys) == 0 and len(unexpected_keys) == 0:
            print("  ✅ Perfect Match!")

    def _load_config_and_model(self):
        print(f"[Init] Loading model config from {self.model_dir}...")
        with open(self.model_dir / "args.json", "r") as f:
            self.config = json.load(f)
            self.args = Namespace(**self.config)

        # 1. Text Encoder (Trainable)
        self.text_encoder = TrainableTextEncoder(
            model_name_or_path=self.youtu_path,
            output_dim=self.args.layer_size,
            dropout=self.args.dropout,
            trust_remote_code=True
        ).to(self.device).eval()
        
        # 2. Text Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.youtu_path, trust_remote_code=True, padding_side="right")
        
        # 3. Fusion Model
        vocab_mod = self.args.modality_dict.copy()
        if '<pad>' not in vocab_mod: vocab_mod['<pad>'] = len(vocab_mod)
        
        self.fusion_model = OmniFusionBlock(
            self.args.num_modalities, 0, 4, 
            self.args.layer_size, self.args.num_layers_fus, 
            self.args.num_experts, self.args.num_routers, 
            self.args.top_k, self.args.nhead, self.args.dropout, 
            vocab_mod=vocab_mod, device=self.device
        ).to(self.device).eval()

        # 4. Load Weights
        print("[Init] Loading weights...")
        ckpts = list(self.model_dir.glob(f"model_e{self.model_epoch}.pt"))
        if not ckpts:
            raise FileNotFoundError(f"Checkpoint for epoch {self.model_epoch} not found.")
        best_model_path = ckpts[0]
        
        print(f"   -> Loading from: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location='cpu')

        def remove_prefix(state_dict):
            return {k.replace("module.", ""): v for k, v in state_dict.items()}

        fusion_sd = remove_prefix(checkpoint['fusion_model'])
        # 使用 Debug 加载器
        self._load_state_dict_with_debug(self.fusion_model, fusion_sd, "FusionModel")
        
        # 加载 Text LoRA
        if 'TEXT' in checkpoint['encoder_dict']:
            print("   -> Loading Text Encoder weights (with LoRA)...")
            text_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            lora_config = LoraConfig(r=self.args.lora_rank, lora_alpha=self.args.lora_alpha, target_modules=text_target_modules, lora_dropout=self.args.dropout, bias="none", modules_to_save=["projection"])
            self.text_encoder = get_peft_model(self.text_encoder, lora_config)
            
            text_sd = remove_prefix(checkpoint['encoder_dict']['TEXT'])
            
            # ========== 【核心修改开始：修复 Key 不匹配】 ==========
            new_text_sd = {}
            fixed_count = 0
            for k, v in text_sd.items():
                # 情况 1: original_0 -> original_module.0
                if "projection.original_0" in k:
                    new_k = k.replace("projection.original_0", "projection.original_module.0")
                    new_text_sd[new_k] = v
                    fixed_count += 1
                # 情况 2: original_1 -> original_module.1
                elif "projection.original_1" in k:
                    new_k = k.replace("projection.original_1", "projection.original_module.1")
                    new_text_sd[new_k] = v
                    fixed_count += 1
                else:
                    new_text_sd[k] = v
            
            if fixed_count > 0:
                print(f"   -> [Fix] Renamed {fixed_count} mismatched keys (original_X -> original_module.X)")
            text_sd = new_text_sd
            # ========== 【核心修改结束】 ==========

            # 使用 Debug 加载器
            self._load_state_dict_with_debug(self.text_encoder, text_sd, "TextEncoder(LoRA)")

    def _load_or_generate_text_cache(self, split_json_path):
        """
        检查 TEXT 模态的 h5ad 缓存。
        如果存在，直接加载。
        如果不存在，读取 JSON -> Filter Valid IDs -> Encode -> Save .h5ad
        """
        # 构造缓存文件名 (与 Omics 保持一致)
        cache_filename = f"tcga_database_TEXT_{self.folder_name}_model_e{self.model_epoch}_youtu_omics_frozen_woproject.h5ad"
        cache_path = self.database_dir / cache_filename

        # === Case 1: 缓存存在 ===
        if cache_path.exists():
            print(f"[Cache] Found Text Cache: {cache_path}")
            print("   -> Loading from disk (backed mode)...")
            adata = sc.read_h5ad(cache_path, backed='r')
            return adata.obs.index.to_numpy(), adata, True # is_z = True

        # === Case 2: 缓存不存在，生成 ===
        print(f"[Cache] Text Cache NOT found at {cache_path}")
        print("   -> Generating Text Embeddings for Valid Set...")

        # 1. 加载 Valid IDs
        with open(split_json_path, 'r') as f:
            split_data = json.load(f)
        valid_ids_set = set(split_data['valid_ids'])

        # 2. 加载原始 Text JSON
        with open(TEXT_JSON_PATH, 'r') as f:
            raw_data = json.load(f)
        
        # 3. 筛选 Valid Data
        valid_ids_list = []
        valid_texts_list = []
        for pid, text in raw_data.items():
            if pid in valid_ids_set:
                valid_ids_list.append(pid)
                valid_texts_list.append(text)
        
        if len(valid_ids_list) == 0:
            raise ValueError("No valid text samples found matching split file.")
        
        print(f"   -> Processing {len(valid_ids_list)} valid text samples...")

        # 4. Batch Encoding
        z_embeddings = []
        batch_size = 32
        
        with torch.no_grad():
            for i in tqdm(range(0, len(valid_texts_list), batch_size), desc="Encoding Text"):
                batch_text = valid_texts_list[i : i+batch_size]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_text, padding=True, truncation=True, 
                    max_length=self.args.max_seq_len_text, 
                    return_tensors='pt', 
                    add_special_tokens=True
                ).to(self.device)
                
                # Encode
                encoded_output, _ = self.text_encoder(
                    input_ids=encoded['input_ids'], 
                    attention_mask=encoded['attention_mask']
                )
                
                # Extract & Adapt
                feat_raw = encoded_output.squeeze(1) # (B, D)
                feat_adapted = self.fusion_model.run_adapter('TEXT', feat_raw)
                
                # Project & Normalize
                feat_z = self.fusion_model.simclr.project_q(feat_adapted, 'TEXT')
                feat_z = F.normalize(feat_z, dim=1)
                
                z_embeddings.append(feat_z.cpu().numpy())

        # 5. 合并并保存
        z_embeddings = np.concatenate(z_embeddings, axis=0)
        
        # 创建 AnnData
        obs_df = pd.DataFrame(index=valid_ids_list)
        adata_emb = AnnData(X=z_embeddings, obs=obs_df)
        
        print(f"   -> Saving Text Embeddings to {cache_path}...")
        adata_emb.write(cache_path)
        print("   -> Cache saved.")

        return np.array(valid_ids_list), adata_emb, True

    def _get_ids_and_data(self, modality, split_json_path):
        """
        获取数据的 ID 列表和数据。
        对于 TEXT: 调用缓存逻辑。
        对于 BIO : 直接读取已存在的缓存文件。
        """
        if modality == 'TEXT':
            return self._load_or_generate_text_cache(split_json_path)
        else:
            # 读取 Step 1 生成的 Bio Embedding
            filename = f"tcga_database_{modality}_{self.folder_name}_model_e{self.model_epoch}_youtu_omics_frozen_woproject.h5ad"
            path = self.database_dir / filename
            
            if not os.path.exists(path): 
                raise FileNotFoundError(f"Embedding file not found: {path}. Please run generate_embeddings.py first.")
            
            # 使用 backed 模式读取
            adata = sc.read_h5ad(path, backed='r')
            ids = np.array(adata.obs.index.tolist())
            return ids, adata, True

    def load_and_align_data(self, source_mod, target_mod, split_json_path, 
                            sample_size=None, seed=42):
        print(f"\n[Data] Aligning {source_mod} -> {target_mod}...")

        # 传递 split_json_path 以便 TEXT 生成缓存时使用
        src_ids, src_data, src_is_z = self._get_ids_and_data(source_mod, split_json_path)
        tgt_ids, tgt_data, tgt_is_z = self._get_ids_and_data(target_mod, split_json_path)

        # 2. 确定 Valid Common IDs
        with open(split_json_path, 'r') as f:
            split_data = json.load(f)
        valid_ids_set = set(split_data['valid_ids'])

        # 取三方交集
        common_ids = np.intersect1d(src_ids, tgt_ids)
        final_ids = np.array([pid for pid in common_ids if pid in valid_ids_set])

        if len(final_ids) == 0: 
            raise ValueError("No common valid IDs found! Check paths and split file.")
        print(f"   -> Found {len(final_ids)} valid aligned samples.")

        # # 3. 采样 (可选)
        if SELECT_STRATEGY == "Random":
            if sample_size and len(final_ids) > sample_size:
                np.random.seed(seed)
                final_ids = np.random.choice(final_ids, size=sample_size, replace=False)
        else:
            # 3. 用 KMeans 代替随机采样
            # 如果 source 是 TEXT，准备和 final_ids 对齐的文本列表
            aligned_text_list = None
            if source_mod == 'TEXT' or target_mod == 'TEXT':
                text_dict = self._load_text_dict()
                aligned_text_list = [text_dict.get(pid, "") for pid in final_ids]

            if sample_size is not None:
                if len(final_ids) < sample_size:
                    raise ValueError(
                        f"Aligned valid samples ({len(final_ids)}) < sample_size ({sample_size}), cannot run KMeans."
                    )

                final_ids = self._select_kmeans_anchors(
                    ids=final_ids,
                    src_ids=src_ids,
                    src_data=src_data,
                    n_clusters=sample_size,
                    seed=seed,
                    n_init=KMEANS_N_INIT,
                    aligned_text_list=aligned_text_list
                )
            
        self.final_ids = final_ids

        # 4. 提取数据
        def extract(ids, all_ids, data):
            # 构建索引映射
            idx_map = {pid: i for i, pid in enumerate(all_ids)}
            indices = [idx_map[pid] for pid in ids]
            
            # 无论 Text 还是 Bio，现在 data 都是 AnnData (Z 向量)
            # data 是 backed AnnData，按索引读取磁盘
            matrix = data[indices].X
            if hasattr(matrix, "toarray"): matrix = matrix.toarray()
            return torch.tensor(matrix, dtype=torch.float32)

        print("   -> Extracting aligned data...")
        self.source_input = extract(final_ids, src_ids, src_data)
        self.target_input = extract(final_ids, tgt_ids, tgt_data)
        
        # 关闭 backed file
        if hasattr(src_data, 'file'): src_data.file.close()
        if hasattr(tgt_data, 'file'): tgt_data.file.close()
    
    def compute_csls_sim(self, q_vecs, db_vecs, k=10):
        # ... (保持原样)
        sim_matrix = torch.matmul(q_vecs, db_vecs.T)
        topk_sim_q, _ = torch.topk(sim_matrix, k=k, dim=1)
        r_q = torch.mean(topk_sim_q, dim=1).unsqueeze(1)
        topk_sim_db, _ = torch.topk(sim_matrix, k=k, dim=0)
        r_db = torch.mean(topk_sim_db, dim=0).unsqueeze(0)
        return 2 * sim_matrix - r_q - r_db

    def evaluate(self, source_mod, target_mod, k_list=[1, 5, 10, 30], sim_metric='csls', csls_k=10):
        """
        sim_metric: 'cosine', 'csls', or 'zscore'
        csls_k: CSLS 算法中的 k 值，仅当 sim_metric='csls' 时有效
        """
        print(f"\n[Evaluation] {source_mod} -> {target_mod} | Metric: {sim_metric}")
        
        # 放到 Device
        z_src = self.source_input.to(self.device)
        z_tgt = self.target_input.to(self.device)
        print(f"  [DEBUG] z_src CLS STD: {z_src.std(dim=0).mean().item():.6f}")
        print(f"  [DEBUG] z_tgt CLS STD: {z_tgt.std(dim=0).mean().item():.6f}")
        
        print("   -> Computing similarity matrix...")
        
        if sim_metric == 'csls':
            # === 方式 1: CSLS ===
            sim_matrix = self.compute_csls_sim(z_src, z_tgt, k=csls_k)
            
        elif sim_metric == 'zscore':
            # === 方式 2: Z-Score 标准化 (去除 Hubness) ===
            # 先算原始相似度
            logits_full = torch.matmul(z_src, z_tgt.T)
            
            # 在 Target 维度 (dim=0) 上进行标准化
            # 这里的 dim=0 意味着对每一列（每个 Target 样本）计算它与所有 Query 的相似度分布
            # 目的是让容易被检索到的 Target (Hubs) 的分数降下来
            mean_per_target = logits_full.mean(dim=0, keepdim=True)
            std_per_target = logits_full.std(dim=0, keepdim=True)
            
            sim_matrix = (logits_full - mean_per_target) / (std_per_target + 1e-6)
            
        else:
            # === 方式 3: 原始 Cosine (Default) ===
            sim_matrix = torch.matmul(z_src, z_tgt.T)
        
        # === 评估指标 ===
        n = sim_matrix.shape[0]
        metrics = {}
        ranks = []
        
        sim_np = sim_matrix.cpu().numpy()
        for i in range(n):
            # 对角线上的值是 Ground Truth (Correct Pair) 的得分
            target_score = sim_np[i, i]
            
            # 计算有多少个样本的分数比 Ground Truth 高
            # (sim_np[i] > target_score) 得到的是 boolean 数组
            rank = (sim_np[i] > target_score).sum() + 1
            ranks.append(rank)
        
        ranks = np.array(ranks)
        for k in k_list:
            metrics[f"R@{k}"] = (ranks <= k).mean()
        metrics["MeanR"] = ranks.mean()
        metrics["MedR"] = np.median(ranks)
        
        print("-" * 30)
        print(f"Sample Size: {n}")
        print(json.dumps(metrics, indent=4))
        print("-" * 30)

if __name__ == "__main__":
    evaluator = RetrievalEvaluator(
        model_dir=MODEL_DIR, 
        device=DEVICE, 
        youtu_path=YOUTU_PATH, 
        model_epoch=EPOCH,
        folder_name=FOLDER_NAME,
        database_dir=DATABASE_DIR
    )
    
    evaluator.load_and_align_data(
        SOURCE_MODALITY, TARGET_MODALITY, SPLIT_JSON_PATH, 
        sample_size=SAMPLE_SIZE
    )
    
    evaluator.evaluate(SOURCE_MODALITY, TARGET_MODALITY)
