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
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# --- Hotfix ---
if not hasattr(transformers.utils, "LossKwargs"):
    class LossKwargs: pass
    transformers.utils.LossKwargs = LossKwargs

warnings.filterwarnings("ignore")
sys.path.insert(0, "./") 
sys.path.append("../")

# 导入 PEFT
from peft import LoraConfig, PeftModel, get_peft_model
# 导入 omix 模块
from omix.model.omni_fusion import (
    OmniFusionBlock, TrainableTextEncoder)
from omix.model.model_multi import PerformerModel
from omix.preprocess_bulk import Preprocessor
from omix.tokenizer import tokenize_and_pad_batch
from omix.tokenizer.gene_tokenizer import GeneVocab


# ==============================================================================
# Inference Class (保持不变)
# ==============================================================================
class CellWhispererInference:
    def __init__(self, model_dir, device, youtu_path):
        self.device = device
        self.model_dir = Path(model_dir)
        self.youtu_path = youtu_path
        
        print(f"[Config] Loading from {self.model_dir}...")
        with open(self.model_dir / "args.json", "r") as f:
            self.args = Namespace(**json.load(f))
            
        # Load Vocab
        vocab_file = self.model_dir / "vocab_rna.json"
        self.vocab = GeneVocab.from_file(vocab_file)
        for s in [self.args.pad_token, "<cls>", "<eoc>"]:
            if s not in self.vocab: self.vocab.append_token(s)
        
        # 初始化 Tokenizer
        print(f"[Tokenizer] Loading from {youtu_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(youtu_path, trust_remote_code=True, padding_side="right")
        # self.query_instruction = "Instruction: You are a medical expert. Given a search clinical text, retrieve the most related omic embedding. Otherwise, your will be punished. \nQuery:"
        
        self._build_architecture()
        self._load_weights()
        
    def _build_architecture(self):
        # 1. RNA Encoder
        self.rna_encoder = PerformerModel(
            ntoken=len(self.vocab), d_model=self.args.layer_size, nhead=self.args.nhead,
            d_hid=self.args.layer_size, nlayers=self.args.nlayers, vocab=self.vocab,
            pad_token=self.args.pad_token, pad_value=self.args.pad_value,
            n_input_bins=self.args.n_bins + 2, cell_emb_style="cls",input_emb_style="category",
            feature_redraw_interval=None, auto_check_redraw=False,
        ).to(self.device).eval()

        # RNA LoRA Check
        use_lora = getattr(self.args, 'use_LoRA', False)
        text_only = getattr(self.args, 'text_only_lora', False)

        if use_lora and not text_only:
            print("[RNA] Applying LoRA...")
            omics_lora_config = LoraConfig(r=self.args.lora_rank, lora_alpha=self.args.lora_alpha, target_modules=["to_q", "to_v"], lora_dropout=self.args.dropout, bias="none")
            self.rna_encoder = get_peft_model(self.rna_encoder, omics_lora_config)
        
        # 2. Text Encoder
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
        
        # 3. Fusion Model
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
            return {k.replace("module.", ""): v for k, v in state_dict.items()}

        if 'RNA' in checkpoint['encoder_dict']:
            self.rna_encoder.load_state_dict(clean_state_dict(checkpoint['encoder_dict']['RNA']), strict=False)
        if 'TEXT' in checkpoint['encoder_dict']:
            self.text_encoder.load_state_dict(clean_state_dict(checkpoint['encoder_dict']['TEXT']), strict=False)
        self.fusion_model.load_state_dict(clean_state_dict(checkpoint['fusion_model']), strict=False)

    def encode_rna(self, adata, batch_size=64):
        print("\n[Inference] Encoding RNA (H-space / Pre-Projector)...")
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
        embeddings_woproject = []
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
                embeddings_woproject.append(feat_h_cls.cpu())

                feat_z = self.fusion_model.simclr.project_q(feat_h_cls, 'RNA') # Project Q
                embeddings.append(F.normalize(feat_z, dim=1).cpu())

                # embeddings.append(F.normalize(feat_h_cls, dim=1).cpu())
                
        return torch.cat(embeddings, dim=0), torch.cat(embeddings_woproject, dim=0)

    def encode_text(self, text_list, batch_size=32):
        print(f"\n[Inference] Encoding {len(text_list)} labels (H-space)...")
        # processed_texts = [f"{self.query_instruction}{t}" for t in text_list]
        processed_texts = text_list
        
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(processed_texts), batch_size)):
                batch_raw = processed_texts[i : i+batch_size]
                
                encoded = self.tokenizer(
                    batch_raw, padding='max_length', truncation=True, 
                    max_length=1024, return_tensors='pt', add_special_tokens=True
                ).to(self.device)
                
                feat_raw, _ = self.text_encoder(
                    input_ids=encoded['input_ids'], 
                    attention_mask=encoded['attention_mask']
                )
                
                feat_adapted = self.fusion_model.run_adapter('TEXT', feat_raw.squeeze(1))
                feat_z = self.fusion_model.simclr.project_q(feat_adapted, 'TEXT')
                embeddings.append(F.normalize(feat_z, dim=1).cpu())

                # embeddings.append(F.normalize(feat_adapted, dim=1).cpu())
                
        return torch.cat(embeddings, dim=0)

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

# ==============================================================================
# Helper: Zero-Shot Classification & Metrics
# ==============================================================================
def evaluate_zero_shot(cw_model, rna_embeddings, adata, label_col, save_dir=None, methods='CSLS'):
    """
    methods = 'CSLS, Zscore'
    执行 Zero-shot 分类评估，并保存预测结果。
    """
    print(f"\n>>> Evaluating Zero-shot Classification for: {label_col}")
    
    # 1. 确定标签集合
    valid_mask = adata.obs[label_col].notna() & (adata.obs[label_col] != 'nan')
    true_labels_subset = adata.obs[label_col][valid_mask].astype(str).values
    unique_labels = sorted(list(set(true_labels_subset)))
    
    print(f"Total labeled samples: {len(true_labels_subset)}")
    print(f"Unique classes: {len(unique_labels)}")
    
    # 2. 构造 Prompts
    prompts = []
    for label in unique_labels:
        if label_col in ['Disease','Disease_subtype']:
            if label == 'Healthy':
                p = "This is a healthy sample, without any disease."
            else:
                p = f"- Patient diagnosed with {label}."
        else: # Tissue
            p = f"The tissue type of this sample: {label}."
        prompts.append(p)
    
    # 3. Encode Labels
    label_embeddings = cw_model.encode_text(prompts) 
    
    # =====================================================
    #  Part A: 计算全量预测
    # =====================================================
    # 【修正】 Query = RNA, DB = Labels
    # 输出形状: (N_cells, N_classes)
    if methods == 'CSLS':
        logits_full = compute_csls_sim(
            rna_embeddings.to(cw_model.device), 
            label_embeddings.to(cw_model.device)
        ).cpu()
        pred_indices = torch.argmax(logits_full, dim=1).numpy()

    else:
        logits_full = torch.matmul(
            rna_embeddings.to(cw_model.device), 
            label_embeddings.to(cw_model.device).T
        ).cpu()
        mean_col = logits_full.mean(dim=0, keepdim=True)
        std_col = logits_full.std(dim=0, keepdim=True)
        logits_col_norm = (logits_full - mean_col) / (std_col + 1e-6)
        mean_row = logits_col_norm.mean(dim=1, keepdim=True)
        std_row = logits_col_norm.std(dim=1, keepdim=True)
        logits_norm = (logits_col_norm - mean_row) / (std_row + 1e-6)
        pred_indices = torch.argmax(logits_norm, dim=1).numpy()
    
    # 【修正】 在 dim=1 (Labels维度) 上找最大值
    
    pred_labels = [unique_labels[i] for i in pred_indices]
    
    # 保存预测结果到 adata.obs
    pred_col_name = f'pred_{label_col}'
    adata.obs[pred_col_name] = pred_labels
    print(f"✅ Predicted labels saved to adata.obs['{pred_col_name}']")

    # =====================================================
    #  Part B: 计算指标
    # =====================================================
    # 取出有真实标签的子集
    pred_labels_subset = np.array(pred_labels)[valid_mask.values]
    
    acc = accuracy_score(true_labels_subset, pred_labels_subset)
    print(f"✅ Accuracy (Subset): {acc:.4f}")
    
    try:
        lb = LabelBinarizer()
        lb.fit(unique_labels)
        y_true_onehot = lb.transform(true_labels_subset)
        
        # 【修正】 Logits 已经是 (N_cells, N_classes) 了，直接切片即可，无需转置
        probs_subset = logits_full.numpy()[valid_mask.values, :] 
        
        auroc = roc_auc_score(y_true_onehot, probs_subset, multi_class='ovr', average='macro')
        print(f"✅ AUROC (Macro): {auroc:.4f}")
    except Exception as e:
        print(f"⚠️ AUROC calculation failed: {e}")
        auroc = None
    
    if save_dir:
        print(f"Calculating global similarity matrix with Double Z-score normalization...")
        
        if methods == 'CSLS':
            sim_matrix = compute_csls_sim(
                label_embeddings.to(cw_model.device), 
                rna_embeddings.to(cw_model.device)
            ).cpu()
            sim_matrix_np = sim_matrix.numpy()
            df_sim = pd.DataFrame(sim_matrix_np, index=unique_labels, columns=adata.obs_names)
            
            # 5. 保存
            os.makedirs(save_dir, exist_ok=True)
            filename = f"disease_similarity_matrix_CSLS_{label_col}.csv" # 改个名区分一下
            save_path = os.path.join(save_dir, filename)
            df_sim.to_csv(save_path)
            print(f"💾 CSLS Similarity matrix saved to: {save_path}")
            print(f"   Shape: {df_sim.shape} (Rows=Labels, Cols=Cells)")

        else:
            logits_full = torch.matmul(
                rna_embeddings.to(cw_model.device), 
                label_embeddings.to(cw_model.device).T
            ).cpu() 
            mean_col = logits_full.mean(dim=0, keepdim=True)
            std_col = logits_full.std(dim=0, keepdim=True)
            logits_col_norm = (logits_full - mean_col) / (std_col + 1e-6)
            mean_row = logits_col_norm.mean(dim=1, keepdim=True)
            std_row = logits_col_norm.std(dim=1, keepdim=True)
            logits_double_norm = (logits_col_norm - mean_row) / (std_row + 1e-6)
            sim_matrix_np = logits_double_norm.numpy().T 
            
            # 5. 创建 DataFrame
            df_sim = pd.DataFrame(sim_matrix_np, index=unique_labels, columns=adata.obs_names)
            
            # 5. 保存
            os.makedirs(save_dir, exist_ok=True)
            filename = f"disease_similarity_matrix_Zscored_{label_col}.csv" # 改个名区分一下
            save_path = os.path.join(save_dir, filename)
            df_sim.to_csv(save_path)
            print(f"💾 Z-scored Similarity matrix saved to: {save_path}")
            print(f"   Shape: {df_sim.shape} (Rows=Labels, Cols=Cells)")


# ==============================================================================
# Main
# ==============================================================================
CW_MODEL_DIR = "../pretrain/save/omix_t_pretrain"
YOUTU_PATH = "./omix/Youtu_embedding"
RNA_FILE = '../../data/cellwhisper/human_disease/human_disease_tpm_log1p_filtered.h5ad'
save_dir = "../../data/cellwhisper/human_disease_20260115"
methods = "ZSCORE" # CSLS, ZSCORE
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    # 1. 初始化模型
    cw_model = CellWhispererInference(CW_MODEL_DIR, DEVICE, YOUTU_PATH)
    
    # 2. 加载 RNA 数据
    print("\nLoading RNA Data...")
    adata = anndata.read_h5ad(RNA_FILE)
    if adata.obs_names.is_unique is False: adata.obs_names_make_unique()
    
    cache_path = Path(f"{save_dir}/rna_embeddings_cache.pt")
    cache_woproject_path = Path(f"{save_dir}/rna_embeddings_cache_woproject.pt")
    
    print(f"\n[Cache Check] Looking for {cache_path}...")
    z_rna = None
    z_rna_woproject = None

    if cache_path.exists() and cache_woproject_path.exists():
        print("✅ Found cached embeddings. Loading...")
        z_rna = torch.load(cache_path, map_location='cpu')
        
        if z_rna.shape[0] != len(adata):
            print(f"⚠️ Warning: Cache size ({z_rna.shape[0]}) mismatch ({len(adata)}). Re-encoding...")
            z_rna = None # 强制重新计算
    
    if z_rna is None and z_rna_woproject is None:
        print("❌ Encoding RNA (this may take a while)...")
        z_rna, z_rna_woproject = cw_model.encode_rna(adata)
        torch.save(z_rna, cache_path)
        torch.save(z_rna_woproject, cache_woproject_path)
        print(f"💾 Embeddings saved to {cache_path} and {cache_woproject_path}")
    
    # =================================================================
    # 3. 将 Embedding 写入 adata
    # =================================================================
    if isinstance(z_rna, torch.Tensor):
        adata.obsm['X_omix'] = z_rna.cpu().numpy()
    else:
        adata.obsm['X_omix'] = z_rna

    # =================================================================
    # 4. 评估并生成 Label
    # =================================================================
    if 'Disease' in adata.obs.columns:
        evaluate_zero_shot(cw_model, z_rna, adata, 'Disease', save_dir, methods=methods)
    
    if 'Disease_subtype' in adata.obs.columns:
        evaluate_zero_shot(cw_model, z_rna, adata, 'Disease_subtype', save_dir, methods=methods)

    if 'Tissue' in adata.obs.columns:
        evaluate_zero_shot(cw_model, z_rna, adata, 'Tissue', save_dir, methods=methods)

    # =================================================================
    # 5. 保存结果
    # =================================================================
    output_h5ad = f"{save_dir}/rna_embeddings_youtu_omics_frozen_woproject_predicted_labels_{methods}.h5ad"
    
    print(f"\n💾 Saving full adata with predictions and embeddings to {output_h5ad}...")
    adata.write(output_h5ad)
    print("Done.")

if __name__ == "__main__":
    main()
