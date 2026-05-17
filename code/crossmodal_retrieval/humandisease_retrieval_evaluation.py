import json
import os
import sys
import warnings
from argparse import Namespace
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import torch
import transformers.utils
from scipy.sparse import issparse
# --- 导入必要的 SKLEARN 模块 ---
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

# ddd = anndata.read_h5ad('../../data/cellwhisper/human_disease/full_data.h5ad')



# --- 路径设置 ---
sys.path.insert(0, "./") 
sys.path.append("../")

# 导入 PEFT
from peft import LoraConfig, get_peft_model
# 导入 omix 模块
from omix.model.omni_fusion import (
    OmniFusionBlock, TrainableTextEncoder)
from omix.model.model_multi import PerformerModel
from omix.preprocess_bulk import Preprocessor
from omix.tokenizer import tokenize_and_pad_batch
from omix.tokenizer.gene_tokenizer import GeneVocab

# Hotfix
if not hasattr(transformers.utils, "LossKwargs"):
    class LossKwargs: pass
    transformers.utils.LossKwargs = LossKwargs
warnings.filterwarnings("ignore")

# RNA, TEXT
SOURCE_MODALITY = 'RNA'
TARGET_MODALITY = 'TEXT'

# ==============================================================================
# Model Class (保持不变)
# ==============================================================================
class RetrievalInference:
    def __init__(self, model_dir, device, youtu_path):
        self.device = device
        self.model_dir = Path(model_dir)
        self.youtu_path = youtu_path
        
        print(f"[Config] Loading from {self.model_dir}...")
        with open(self.model_dir / "args.json", "r") as f:
            self.args = Namespace(**json.load(f))
            
        vocab_file = self.model_dir / "vocab_rna.json"
        self.vocab = GeneVocab.from_file(vocab_file)
        for s in [self.args.pad_token, "<cls>", "<eoc>"]:
            if s not in self.vocab: self.vocab.append_token(s)
        
        print(f"[Tokenizer] Loading from {youtu_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(youtu_path, trust_remote_code=True, padding_side="right")
        # self.query_instruction = "Instruction: Given a search query, retrieve passages that answer the question \nQuery:"
        
        self._build_architecture()
        self._load_weights()
        
    def _build_architecture(self):
        # 1. RNA Encoder
        self.rna_encoder = PerformerModel(
            ntoken=len(self.vocab), d_model=self.args.layer_size, nhead=self.args.nhead,
            d_hid=self.args.layer_size, nlayers=self.args.nlayers, vocab=self.vocab,
            pad_token=self.args.pad_token, pad_value=self.args.pad_value,
            n_input_bins=self.args.n_bins + 2, cell_emb_style="cls", input_emb_style="category",
            feature_redraw_interval=None, auto_check_redraw=False,
        ).to(self.device).eval()

        # RNA LoRA Check
        use_lora = getattr(self.args, 'use_LoRA', False)
        text_only = getattr(self.args, 'text_only_lora', False)

        if use_lora and not text_only:
            print("[RNA] Applying LoRA...")
            omics_lora_config = LoraConfig(r=self.args.lora_rank, lora_alpha=self.args.lora_alpha, target_modules=["to_q", "to_v"], lora_dropout=self.args.dropout, bias="none")
            self.rna_encoder = get_peft_model(self.rna_encoder, omics_lora_config)
        
        # 2. Text Encoder (Trainable)
        print("[Text] Building TrainableTextEncoder...")
        print(self.device)
        print('opkoko')
        self.text_encoder = TrainableTextEncoder(
            model_name_or_path=self.youtu_path,
            output_dim=self.args.layer_size,
            dropout=self.args.dropout,
            trust_remote_code=True
        ).to(self.device).eval()

        # Text LoRA
        if use_lora:
            print("[Text] Applying LoRA...")
            text_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            lora_config = LoraConfig(r=self.args.lora_rank, lora_alpha=self.args.lora_alpha, target_modules=text_target_modules, lora_dropout=self.args.dropout, bias="none", modules_to_save=["projection"])
            self.text_encoder = get_peft_model(self.text_encoder, lora_config)
        
        # 3. Fusion Model (Adapter + Projector)
        vocab_mod = self.args.modality_dict.copy()
        if '<pad>' not in vocab_mod: vocab_mod['<pad>'] = len(vocab_mod)
        
        self.fusion_model = OmniFusionBlock(
            self.args.num_modalities, 0, 4, self.args.layer_size, 
            self.args.num_layers_fus, self.args.num_experts, self.args.num_routers, 
            self.args.top_k, self.args.nhead, self.args.dropout, vocab_mod=vocab_mod,
            device=self.device
        ).to(self.device).eval()

    def _load_weights(self):
        ckpt_path = self.model_dir / "model_e5.pt"
                 
        print(f"[Weights] Loading from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        def clean_state_dict(state_dict):
            new_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                if "projection.original_" in k and "original_module" not in k:
                    k = k.replace("projection.original_", "projection.original_module.")
                new_dict[k] = v
            return new_dict

        if 'RNA' in checkpoint['encoder_dict']:
            self.rna_encoder.load_state_dict(clean_state_dict(checkpoint['encoder_dict']['RNA']), strict=True)
        
        # =========================================================
        # 🕵️‍♂️ [Step 1] 抓取加载前的 Projection 快照
        # =========================================================
        proj_param_before = None
        proj_name_tracker = None
        
        # 寻找 Text Encoder 的 Projection 层参数
        for name, param in self.text_encoder.named_parameters():
            if "projection" in name and "weight" in name and param.requires_grad:
                proj_param_before = param.data.clone() # 必须 clone!
                proj_name_tracker = name
                break # 取一个样本就够了

        # =========================================================
        # [Step 2] 执行加载
        # =========================================================
        if 'TEXT' in checkpoint['encoder_dict']:
            print("[Text] Loading weights...")
            self.text_encoder.load_state_dict(clean_state_dict(checkpoint['encoder_dict']['TEXT']), strict=True)
            
            # =========================================================
            # 🕵️‍♂️ [Step 3] 验证报告 (Verification Report)
            # =========================================================
            print("\n" + "="*50)
            print("🕵️‍♂️ [Verification Report] Text Encoder Training Status")
            print("="*50)

            # --- A. 验证 LoRA (检查是否非0) ---
            lora_trained = False
            lora_found = False
            
            for name, param in self.text_encoder.named_parameters():
                if "lora_B" in name:
                    lora_found = True
                    # 检查数值是否全为 0
                    if torch.all(param == 0).item():
                        pass # 还是0，没训练到
                    else:
                        lora_trained = True
                        print(f"✅ [LoRA] Status: ACTIVE (Trained)")
                        print(f"   -> Found non-zero weights in: {name}")
                        print(f"   -> Sample values: {param.flatten()[:5].tolist()}")
                        break # 只要发现一层是活的，就认为 LoRA 活了
            
            if not lora_found:
                print("❓ [LoRA] Status: NOT FOUND (Is LoRA config correct?)")
            elif not lora_trained:
                print("💀 [LoRA] Status: DEAD (All lora_B weights are ZERO)")
                print("   -> Cause: Gradient didn't reach LoRA layers during pre-training.")

            print("-" * 50)

            # --- B. 验证 Projection (检查是否变化) ---
            proj_trained = False
            
            if proj_param_before is not None:
                # 获取加载后的参数
                current_params = dict(self.text_encoder.named_parameters())
                proj_param_after = current_params[proj_name_tracker].data
                
                # 计算差异
                diff = (proj_param_before - proj_param_after.to(proj_param_before.device)).abs().sum().item()
                
                if diff > 1e-6:
                    proj_trained = True
                    print(f"✅ [Projection] Status: ACTIVE (Trained)")
                    print(f"   -> Weights changed from random init. Diff: {diff:.6f}")
                    print(f"   -> Layer: {proj_name_tracker}")
                else:
                    print(f"💀 [Projection] Status: INACTIVE (Same as Random Init)")
                    print(f"   -> Diff is 0.0. The weights are identical to initialization.")
            else:
                print("❓ [Projection] Status: UNKNOWN (Could not find projection layer to compare)")

            # --- C. 最终结论 ---
            print("-" * 50)
            if lora_trained:
                print("🎉 FINAL RESULT: Text Encoder is fully adapted (LoRA + Projection).")
            elif proj_trained:
                print("⚠️ FINAL RESULT: Partial Adaptation.")
                print("   - LoRA is frozen (failed).")
                print("   - Projection IS trained.")
                print("   -> Model is working as a 'Frozen BERT + Trainable MLP'.")
            else:
                print("❌ FINAL RESULT: Text Encoder is completely untrainted (Random Projection).")
            print("="*50 + "\n")

        self.fusion_model.load_state_dict(clean_state_dict(checkpoint['fusion_model']), strict=True)

    def encode_rna(self, adata, batch_size=16):
        
        print("\n[Inference] Encoding RNA (Z-space)...")
        preprocessor = Preprocessor(
            use_key="X", filter_gene_by_counts=False, filter_cell_by_counts=False,
            normalize_total=False, log1p=False, 
            binning=self.args.n_bins, result_binned_key="X_binned", repair=False,
        )
        preprocessor(adata)
        
        input_layer_key = "X_binned"
        data_matrix = adata.layers[input_layer_key]
        if issparse(data_matrix): data_matrix = data_matrix.toarray()
        
        var_in_vocab_mask = np.array([g in self.vocab for g in adata.var_names])
        data_matrix = data_matrix[:, var_in_vocab_mask]
        filtered_var_names = adata.var_names[var_in_vocab_mask].tolist()
        gene_ids = np.array(self.vocab(filtered_var_names), dtype=int)
        
        tokenized = tokenize_and_pad_batch(
            data_matrix, gene_ids, max_len=self.args.max_seq_len_rna, 
            vocab=self.vocab, pad_token=self.args.pad_token, 
            pad_value=self.args.pad_value, append_cls=True, include_zero_gene=False
        )
        
        src_tensor = tokenized['genes']
        val_tensor = tokenized['values']
        
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(adata), batch_size)):
                end = min(i + batch_size, len(adata))
                src = src_tensor[i:end].to(self.device)
                val = val_tensor[i:end].to(self.device)
                
                output, _ = self.rna_encoder(
                    src=src, values=val,
                    src_key_padding_mask=src.eq(self.vocab[self.args.pad_token]),
                    return_seq_embedding=True
                )
                
                feat_h_seq = self.fusion_model.run_adapter('RNA', output)
                feat_h_cls = feat_h_seq[:, 0, :]
                
                feat_z = self.fusion_model.simclr.project_q(feat_h_cls, 'RNA') 
                embeddings.append(F.normalize(feat_z, dim=1).cpu())
                
        return torch.cat(embeddings, dim=0)

    def encode_text(self, text_list, batch_size=16):
        print(f"\n[Inference] Encoding {len(text_list)} texts (Z-space)...")
        # processed_texts = [f"{self.query_instruction}{t}" for t in text_list]
        processed_texts = text_list
        
        embeddings_woproject = []
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(processed_texts), batch_size)):
                batch_raw = processed_texts[i : i+batch_size]
                
                encoded = self.tokenizer(
                    batch_raw, padding=True, truncation=True, 
                    max_length=1024, return_tensors='pt', add_special_tokens=True
                ).to(self.device)
                
                feat_raw, _ = self.text_encoder(
                    input_ids=encoded['input_ids'], 
                    attention_mask=encoded['attention_mask']
                )
                
                feat_raw = feat_raw.squeeze(1) if feat_raw.dim() == 3 else feat_raw

                feat_adapted = self.fusion_model.run_adapter('TEXT', feat_raw)
                
                feat_z = self.fusion_model.simclr.project_q(feat_adapted, 'TEXT')
                embeddings.append(F.normalize(feat_z, dim=1).cpu())
                embeddings_woproject.append(F.normalize(feat_adapted, dim=1).cpu())
                
        return torch.cat(embeddings, dim=0), torch.cat(embeddings_woproject, dim=0)

# ==============================================================================
# Main Logic
# ==============================================================================
# CW_MODEL_DIR = "./save/dev_multimodal_SimCLR_withtext_modalitytoken_prealign_cls_moco_youtu-DDP-Nov29-23-20"
CW_MODEL_DIR = "../pretrain/save/omix_t_pretrain"
YOUTU_PATH = "./omix/Youtu_embedding"
RNA_FILE = '../../data/cellwhisper/human_disease/human_disease_tpm_log1p_filtered.h5ad'
# TEXT_FILE = './OMedical/response_bft_all_new.json'
TEXT_FILE = '../../data/cellwhisper/human_disease/response_llama_bft_all.json'
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
K_MEANS_CLUSTERS = 100 # 聚类数
RANDOM_SEED = 42

def compute_csls_sim(q_vecs, db_vecs, k=10):
    # q_vecs: (N_samples, Dim)
    # db_vecs: (N_labels, Dim)
    sim_matrix = torch.matmul(q_vecs, db_vecs.T) # (N_samples, N_labels)
    
    # 针对每个 Sample，找到它和 Label 库中 Top-K 近的
    topk_sim_q, _ = torch.topk(sim_matrix, k=min(k, sim_matrix.size(1)), dim=1)
    r_q = torch.mean(topk_sim_q, dim=1).unsqueeze(1)
    
    # 针对每个 Label，找到它和 Sample 库中 Top-K 近的
    topk_sim_db, _ = torch.topk(sim_matrix, k=min(k, sim_matrix.size(0)), dim=0)
    r_db = torch.mean(topk_sim_db, dim=0).unsqueeze(0)
    
    return 2 * sim_matrix - r_q - r_db

def extract_final_response(json_str):
    """
    鲁棒的提取函数：能处理完整JSON，也能处理截断的JSON
    """
    import re
    if not json_str:
        return ""
        
    cleaned_str = json_str.strip()
    
    # --- 方法 1: 尝试标准 JSON 解析 ---
    try:
        data = json.loads(cleaned_str)
        return data.get("2. Final Response", "")
    except json.JSONDecodeError:
        pass # 解析失败，进入下一步

    # --- 方法 2: 尝试修复截断的 JSON (补全 '}') ---
    # 你的数据中 SRX085334 就属于这种情况
    try:
        data = json.loads(cleaned_str + "}")
        return data.get("2. Final Response", "")
    except json.JSONDecodeError:
        pass

    # --- 方法 3: 正则表达式暴力提取 (保底方案) ---
    # 匹配 "2. Final Response": 后面的双引号内容
    # 解释: 找到 Key，忽略冒号空格，捕获双引号内的所有内容(非贪婪)，直到遇到下一个双引号或字符串结束
    pattern = r'"2\. Final Response":\s*"(.*?)(?:"\s*\}|$)'
    match = re.search(pattern, cleaned_str, re.DOTALL)
    
    if match:
        # 这里需要处理一下转义字符，比如把 \" 变回 "
        # 使用 codecs.decode 或者简单的 replace
        raw_text = match.group(1)
        return raw_text.replace('\\"', '"').replace('\\n', '\n')
    
    return "ERROR: Unable to extract"


def main():
    # 1. 初始化模型
    model = RetrievalInference(CW_MODEL_DIR, DEVICE, YOUTU_PATH)
    
    # 2. 加载 RNA 数据 (H5AD)
    print("\n[Data] Loading RNA Data...")
    adata = anndata.read_h5ad(RNA_FILE)
    if adata.obs_names.is_unique is False: 
        adata.obs_names_make_unique()
    
    # --- RNA 缓存逻辑 ---
    rna_file_path = Path(RNA_FILE)
    rna_cache_path = rna_file_path.with_name(rna_file_path.stem + "_z_embeddings_cache_0108.pt")
    
    z_rna_full = None
    # if rna_cache_path.exists():
    #     print(f"✅ Found cached RNA embeddings at {rna_cache_path}. Loading...")
    #     z_rna_full = torch.load(rna_cache_path, map_location='cpu')
        
    #     if z_rna_full.shape[0] != len(adata):
    #         print(f"⚠️ RNA Cache mismatch. Re-encoding...")
    #         z_rna_full = None
            
    if z_rna_full is None:
        print("❌ Generating new RNA embeddings...")
        z_rna_full = model.encode_rna(adata)
        torch.save(z_rna_full, rna_cache_path)
        print(f"💾 RNA Embeddings saved to {rna_cache_path}")

    # 3. 加载 Text 数据 (JSON)
    print("\n[Data] Loading Text Data...")
    with open(TEXT_FILE, 'r') as f:
        text_data_dict = json.load(f)
    keep_response_only = False

    # 4. 数据对齐 (Alignment)
    rna_ids = adata.obs_names.tolist()
    text_ids = list(text_data_dict.keys())
    
    # 确定共同的样本 ID，并排序以确保对齐
    common_ids = sorted(list(set(rna_ids) & set(text_ids)))
    print(f"\n[Alignment] Found {len(common_ids)} common samples between RNA and Text.")
    
    if len(common_ids) < K_MEANS_CLUSTERS:
        print(f"Error: Common samples ({len(common_ids)}) is less than K-Means clusters ({K_MEANS_CLUSTERS}). Cannot proceed with clustering.")
        return

    # 提取对齐后的 Text 列表
    text_list_aligned = [text_data_dict[pid] for pid in common_ids]

    if keep_response_only:
        # --- TEXT 缓存逻辑 (新增) ---
        text_cache_path = Path(TEXT_FILE).with_name(Path(TEXT_FILE).stem + "_z_embeddings_cache_responseonly_0108.pt")
    else:
        text_cache_path = Path(TEXT_FILE).with_name(Path(TEXT_FILE).stem + "_z_embeddings_cache_0108.pt")
    z_text_aligned = None
    z_text_aligned_woproject = None

    # if text_cache_path.exists():
    #     print(f"✅ Found cached Text embeddings at {text_cache_path}. Loading...")
    #     try:
    #         loaded_dict = torch.load(text_cache_path, map_location='cpu')
            
    #         # [修正2] 检查缓存是否包含所有需要的 Key (防止读取旧版本缓存报错)
    #         required_keys = ['ids', 'embeddings', 'embeddings_woproject']
    #         if not all(k in loaded_dict for k in required_keys):
    #             raise ValueError("Cache file is missing keys (likely old version). Re-encoding.")

    #         # 检查缓存是否包含所有需要的 common_ids
    #         cached_ids = loaded_dict['ids']
    #         if len(set(common_ids) - set(cached_ids)) == 0:
    #             print("✅ Cache is complete. Aligning cached embeddings...")
                
    #             # 创建缓存 ID 到索引的映射
    #             cached_id_to_idx = {pid: i for i, pid in enumerate(cached_ids)}
                
    #             # 找到 common_ids 在缓存中的索引
    #             text_indices = [cached_id_to_idx[pid] for pid in common_ids]
    #             z_text_aligned = loaded_dict['embeddings'][text_indices]
    #             z_text_aligned_woproject = loaded_dict['embeddings_woproject'][text_indices]
    #         else:
    #             print(f"⚠️ Text Cache mismatch. Need {len(common_ids)} ids, but cache missing some.")
    #             z_text_aligned = None # 显式确保它是 None
                
    #     except Exception as e:
    #         print(f"⚠️ Error loading cache or cache invalid: {e}")
    #         print("   -> Will re-encode text.")
    #         z_text_aligned = None
    #         z_text_aligned_woproject = None

    # 判断 z_text_aligned 是否为空
    if z_text_aligned is None:
        # 如果缓存缺失或不匹配，则重新编码
        z_text_aligned, z_text_aligned_woproject = model.encode_text(text_list_aligned)
        
        # 准备要保存的完整数据 (ID + Embeddings)
        save_data = {
            'ids': common_ids, 
            'embeddings': z_text_aligned.cpu(), 
            'embeddings_woproject': z_text_aligned_woproject.cpu()
        }
        torch.save(save_data, text_cache_path)
        print(f"💾 Text Embeddings saved to {text_cache_path}")

    # 5. 提取对齐后的 RNA Embedding (如果之前缓存成功，这里只需要重新提取索引)
    rna_id_to_idx = {pid: i for i, pid in enumerate(rna_ids)}
    rna_indices = [rna_id_to_idx[pid] for pid in common_ids]
    z_rna_aligned = z_rna_full[rna_indices]
    
    # =================================================================
    # 6. K-Means 聚类和采样 (保持不变)
    # =================================================================
    print(f"\n[Sampling] Running K-Means clustering ({K_MEANS_CLUSTERS} clusters) for prototype selection...")
    
    # K-Means 期望 numpy 数组
    cand_text_embs = z_text_aligned.numpy()
    cand_text_embs_woproject = z_text_aligned_woproject.numpy()
    cand_rna_embs = z_rna_aligned.numpy()
    
    # # 初始化 K-Means
    kmeans = KMeans(n_clusters=K_MEANS_CLUSTERS, random_state=RANDOM_SEED, n_init=10)
    # 拟合聚类（在文本嵌入空间
    kmeans.fit(cand_text_embs)
    # 找到每个簇中心最近的样本的索引 (closest_idx_in_pool 是在 cand_text_embs 中的索引)
    closest_idx_in_pool, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, cand_text_embs)
    print(f"\n[Analysis] Extracting details for the {len(closest_idx_in_pool)} selected prototypes...")

    # print(f"\n[Sampling] Randomly selecting {K_MEANS_CLUSTERS} samples...")
    # total_samples = len(common_ids)
    # n_select = min(K_MEANS_CLUSTERS, total_samples)
    # np.random.seed(RANDOM_SEED) 
    # closest_idx_in_pool = np.random.choice(total_samples, size=n_select, replace=False)

    prototype_data = []
    for cluster_idx, pool_idx in enumerate(closest_idx_in_pool):
        # pool_idx 是 cand_text_embs 中的索引，
        # 而 cand_text_embs 来自 z_text_aligned，
        # 所以它直接对应 common_ids 和 text_list_aligned 的下标。
        
        s_id = common_ids[pool_idx]
        s_text = text_list_aligned[pool_idx]
        
        prototype_data.append({
            "Cluster_ID": cluster_idx,       # 属于第几个聚类中心
            "Sample_Index": pool_idx,        # 在原始列表中的位置
            "Sample_ID": s_id,               # 样本ID
            "Text_Content": s_text           # 完整文本
        })

    # 转为 DataFrame 方便操作
    df_prototypes = pd.DataFrame(prototype_data)

    # 1. 终端打印预览 (只显示 ID 和 文本前 50 个字符)
    print("\n--- Prototype Preview (Top 5) ---")
    preview_df = df_prototypes.copy()
    preview_df['Text_Content'] = preview_df['Text_Content'].apply(lambda x: x[:50] + "..." if len(x) > 50 else x)
    print(preview_df[['Cluster_ID', 'Sample_ID', 'Text_Content']].head(5))

    # 2. 保存到 CSV (推荐)
    save_path = "../../data/cellwhisper/human_disease/kmeans_prototypes_100.csv"
    df_prototypes.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ Full prototype list saved to: {save_path}")
    print("   (Check this file to read the full text content)")

    
    # 提取采样的 100 个 RNA 和 Text 嵌入
    z_rna_sampled = z_rna_aligned[closest_idx_in_pool]
    z_text_sampled = z_text_aligned[closest_idx_in_pool]
    
    print(f"✅ Selected {len(z_rna_sampled)} prototype pairs for final retrieval evaluation.")

    # 7. 计算 Recall@k (保持不变)
    print("\n[Evaluation] Calculating Recall@k on sampled prototypes...")
    if SOURCE_MODALITY == 'RNA':
        z_query = z_rna_sampled.to(DEVICE) # RNA (Query)
        z_db = z_text_sampled.to(DEVICE)   # Text (Database)
    else:
        z_query = z_text_sampled.to(DEVICE) # Text (Query)
        z_db = z_rna_sampled.to(DEVICE)   # RNA (Database)
    
    # 计算相似度矩阵
    # logits = torch.matmul(z_query, z_db.T) 

    # sim_matrix = logits
    
    # mean_col = logits.mean(dim=0, keepdim=True)
    # std_col = logits.std(dim=0, keepdim=True)
    # logits_col_norm = (logits - mean_col) / (std_col + 1e-6)
    # sim_matrix = logits_col_norm
    # mean_row = logits_col_norm.mean(dim=1, keepdim=True)
    # std_row = logits_col_norm.std(dim=1, keepdim=True)
    # sim_matrix = (logits_col_norm - mean_row) / (std_row + 1e-6)
    

    sim_matrix = compute_csls_sim(z_query, z_db)
    
    # 计算 Metrics
    n = sim_matrix.shape[0]
    
    # Recall Calculation
    ranks = []
    sim_np = sim_matrix.cpu().numpy()
    
    for i in range(n):
        true_score = sim_np[i, i]
        rank = (sim_np[i] > true_score).sum() + 1
        ranks.append(rank)
        
    ranks = np.array(ranks)
    metrics = {}
    for k in [1, 5, 10, 30]:
        metrics[f"R@{k}"] = (ranks <= k).mean()
        
    metrics["MeanR"] = ranks.mean()
    metrics["MedR"] = np.median(ranks)
    
    print("\n>>> Retrieval Result (K-Means Sampled Prototypes):")
    print(f"Total Samples (N): {n}")
    print(metrics)

if __name__ == "__main__":
    main()
