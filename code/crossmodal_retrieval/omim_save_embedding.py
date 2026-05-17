
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

RNA_FILE = '../../data/cellwhisper/human_disease/human_disease_tpm_log1p_filtered.h5ad'
adata = anndata.read_h5ad(RNA_FILE)
print('ok')


# ==============================================================================
# 2. Inference Class (整合所有模态)
# ==============================================================================
class CellWhispererInference:
    def __init__(self, model_dir, device, youtu_path, model_epoch):
        self.device = device
        self.model_dir = Path(model_dir)
        self.model_epoch = model_epoch
        self.youtu_path = youtu_path
        
        print(f"[Config] Loading from {self.model_dir}...")
        with open(self.model_dir / "args.json", "r") as f:
            self.args = Namespace(**json.load(f))
            
        # Load Vocab
        vocab_file = self.model_dir / "vocab_rna.json"
        self.vocab = GeneVocab.from_file(vocab_file)
        for s in [self.args.pad_token, "<cls>", "<eoc>"]:
            if s not in self.vocab: self.vocab.append_token(s)
        
        # 初始化 Tokenizer (用于 Text)
        print(f"[Tokenizer] Loading from {youtu_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(youtu_path, trust_remote_code=True, padding_side="right")
        
        self._build_architecture()
        self._load_weights()
        
    def _build_architecture(self):
        # 1. RNA Encoder (Performer)
        self.rna_encoder = PerformerModel(
            ntoken=len(self.vocab), d_model=self.args.layer_size, nhead=self.args.nhead,
            d_hid=self.args.layer_size, nlayers=self.args.nlayers, vocab=self.vocab,
            pad_token=self.args.pad_token, pad_value=self.args.pad_value,
            n_input_bins=self.args.n_bins + 2, cell_emb_style="cls",input_emb_style="category",feature_redraw_interval=None,
            auto_check_redraw=False,
        ).to(self.device).eval()

        use_lora = getattr(self.args, 'use_LoRA', False)
        text_only = getattr(self.args, 'text_only_lora', False)

        if use_lora and not text_only:
            print("[RNA] Applying LoRA (Full-Modality LoRA Mode)...")
            omics_lora_config = LoraConfig(
                r=self.args.lora_rank, 
                lora_alpha=self.args.lora_alpha, 
                target_modules=["to_q", "to_v"], # Performer 的 target
                lora_dropout=self.args.dropout, 
                bias="none"
            )
            self.rna_encoder = get_peft_model(self.rna_encoder, omics_lora_config)
        else:
            print("[RNA] Using Frozen/Standard Encoder (No LoRA).")
        
        # 2. Text Encoder (TrainableTextEncoder)
        self.text_encoder = TrainableTextEncoder(
            model_name_or_path=self.youtu_path,
            output_dim=self.args.layer_size,
            dropout=self.args.dropout,
            trust_remote_code=True
        ).to(self.device).eval()

        # 3. Text LoRA Wrapper
        # 如果训练时使用了 LoRA，推理时必须加回来
        if use_lora:
            print("[Text] Applying LoRA...")
            text_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            # 兼容旧配置：有些旧 checkpoint 可能只训了 q,v，这里最好根据 args 动态调整，或者默认全量
            # 如果训练脚本里写死全量，这里也写死全量
            lora_config = LoraConfig(
                r=self.args.lora_rank, 
                lora_alpha=self.args.lora_alpha, 
                target_modules=text_target_modules, 
                lora_dropout=self.args.dropout, 
                bias="none", 
                modules_to_save=["projection"]
            )
            self.text_encoder = get_peft_model(self.text_encoder, lora_config)
        
        # 4. Fusion Model (Adapter + Projector)
        vocab_mod = self.args.modality_dict.copy()
        if '<pad>' not in vocab_mod: vocab_mod['<pad>'] = len(vocab_mod)
        
        self.fusion_model = OmniFusionBlock(
            self.args.num_modalities, 0, 4, self.args.layer_size, 
            self.args.num_layers_fus, self.args.num_experts, self.args.num_routers, 
            self.args.top_k, self.args.nhead, self.args.dropout, vocab_mod=vocab_mod,
            device=self.device
        ).to(self.device).eval()

    def _load_weights(self):
        ckpt_path = self.model_dir / self.model_epoch
        print(f"[Weights] Loading from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # --- 辅助函数：清理 DDP 前缀 + 修复 PEFT 命名 ---
        def clean_and_fix_keys(state_dict):
            new_dict = {}
            for k, v in state_dict.items():
                # 1. 去掉 DDP 的 module. 前缀
                k = k.replace("module.", "")
                
                # 2. 修复 PEFT Projection 命名不匹配问题
                # 把 "projection.original_0.weight" -> "projection.original_module.0.weight"
                if "projection.original_" in k:
                    k = k.replace("projection.original_", "projection.original_module.")
                
                new_dict[k] = v
            return new_dict

        # ======================================================================
        # 1. 设置探针 (用于验证)
        # ======================================================================
        probes = {}
        # Text Projection Probe
        for name, param in self.text_encoder.named_parameters():
            if "projection" in name and "weight" in name and "lora" not in name:
                probes['Text_Projection'] = param.data.clone()
                probes['Text_Projection_Name'] = name
                break
        
        # RNA Probe
        for name, param in self.rna_encoder.named_parameters():
             if ("layers.0" in name or "layers.1" in name) and "weight" in name and "norm" not in name:
                probes['RNA_Encoder_Layer'] = param.data.clone()
                probes['RNA_Encoder_Name'] = name
                break

        # ======================================================================
        # 2. 执行加载 (使用修复后的字典)
        # ======================================================================
        
        # Load RNA
        if 'RNA' in checkpoint['encoder_dict']:
            print("📥 Loading RNA Encoder...")
            # RNA 部分通常只需要去前缀
            self.rna_encoder.load_state_dict(clean_and_fix_keys(checkpoint['encoder_dict']['RNA']), strict=True)
            
        # Load Text (关键修正部分)
        if 'TEXT' in checkpoint['encoder_dict']:
            print("📥 Loading Text Encoder...")
            # 这里必须用 strict=False，因为 PEFT 可能还有其他不需要的元数据
            # 但经过 clean_and_fix_keys 后，关键的 Projection 层应该能匹配上了
            fixed_text_dict = clean_and_fix_keys(checkpoint['encoder_dict']['TEXT'])
            
            # 调试打印：确认 key 变没变
            # sample_key = list(fixed_text_dict.keys())[0]
            # print(f"DEBUG Key Example: {sample_key}") 
            
            self.text_encoder.load_state_dict(fixed_text_dict, strict=True)
            
        # Load Fusion
        print("📥 Loading Fusion Model...")
        self.fusion_model.load_state_dict(clean_and_fix_keys(checkpoint['fusion_model']), strict=True)

        # ======================================================================
        # 3. 验证结果
        # ======================================================================
        print("\n" + "-"*50)
        print(f"{'Module':<30} | {'Status':<10}")
        print("-" * 50)

        # Check Text Projection
        if 'Text_Projection' in probes:
            name = probes['Text_Projection_Name']
            curr = dict(self.text_encoder.named_parameters())[name].data
            prev = probes['Text_Projection']
            if not torch.equal(prev, curr):
                print(f"{'Text Projection':<30} | ✅ PASS (Loaded Successfully)")
            else:
                print(f"{'Text Projection':<30} | 🔴 FAIL (Still Random Weight)")
                # 如果还失败，打印一下字典里的 key 看看是不是拼写还有问题
                # print("Available keys in fixed dict:", [k for k in fixed_text_dict.keys() if "projection" in k])

        # Check RNA
        if 'RNA_Encoder_Layer' in probes:
            name = probes['RNA_Encoder_Name']
            curr = dict(self.rna_encoder.named_parameters())[name].data
            prev = probes['RNA_Encoder_Layer']
            if not torch.equal(prev, curr):
                print(f"{'RNA Encoder':<30} | ✅ PASS/MOVED")
            else:
                # 如果是 Frozen 模型，这里没变是正常的
                print(f"{'RNA Encoder':<30} | ⚠️ SAME (Normal if Frozen)")

        print("="*50 + "\n")

    def encode_rna(self, adata, batch_size=16):
        print("\n[Inference] Encoding RNA...")
        # Preprocessing
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
        
        z_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(adata), batch_size)):
                end = min(i + batch_size, len(adata))
                src = src_tensor[i:end].to(self.device)
                val = val_tensor[i:end].to(self.device)
                
                # 1. Encoder
                output, _ = self.rna_encoder(
                    src=src, values=val,
                    src_key_padding_mask=src.eq(self.vocab[self.args.pad_token]),
                    return_seq_embedding=True
                )
                
                # 2. Adapter
                feat_h_seq = self.fusion_model.run_adapter('RNA', output)
                
                # # 3. Projector (CLS Token)
                feat_h_cls = feat_h_seq[:, 0, :]
                # feat_z = self.fusion_model.simclr.project_q(feat_h_cls, 'RNA') # Project Q
                
                # z_embeddings.append(F.normalize(feat_z, dim=1).cpu())
                z_embeddings.append(F.normalize(feat_h_cls, dim=1).cpu())
                
        return torch.cat(z_embeddings, dim=0)

    def encode_text(self, processed_texts, batch_size=32):
        print(f"\n[Inference] Encoding {len(processed_texts)} texts...")
        
        z_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(processed_texts), batch_size)):
                batch_raw = processed_texts[i : i+batch_size]

                # 1. Tokenize
                encoded = self.tokenizer(
                    batch_raw, padding='max_length', truncation=True, 
                    max_length=1024, return_tensors='pt', add_special_tokens=True
                ).to(self.device)
                
                # 2. Text Encoder (BERT + LoRA + Projection)
                # Output: (B, 1, Dim)
                feat_raw, _ = self.text_encoder(
                    input_ids=encoded['input_ids'], 
                    attention_mask=encoded['attention_mask']
                )
                
                # 3. Adapter
                # squeeze 适配 adapter, 再 unsqueeze 回来
                feat_adapted = self.fusion_model.run_adapter('TEXT', feat_raw.squeeze(1))

                # feat_z = self.fusion_model.simclr.project_q(feat_adapted, 'TEXT')
                
                # z_embeddings.append(F.normalize(feat_z, dim=1).cpu())

                z_embeddings.append(F.normalize(feat_adapted.squeeze(1), dim=1).cpu())
                
        return torch.cat(z_embeddings, dim=0)

# ==============================================================================
# 3. 主程序
# ==============================================================================
CW_MODEL_DIR = "../pretrain/save/omix_t_pretrain"
model_epoch = "model_e5.pt"
YOUTU_MODEL_PATH = "./omix/Youtu_embedding"
RNA_FILE = '../../data/cellwhisper/human_disease/human_disease_tpm_log1p_filtered.h5ad'
DISEASE_FILE = '../../data/cellwhisper/human_disease/OMIM_gene_score.json'
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "../../data/cellwhisper/human_disease_20260115"

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    model_name = CW_MODEL_DIR.split('/')[-1] + '_' + model_epoch

    # 1. 初始化
    cw_model = CellWhispererInference(CW_MODEL_DIR, DEVICE, YOUTU_MODEL_PATH, model_epoch=model_epoch)
    

    # 2. RNA Inference
    rna_cache_path = Path(os.path.join(OUTPUT_DIR, f"rna_embeddings_youtu_omics_frozen_{model_name}.h5ad"))
    
    z_rna = None
    if rna_cache_path.exists():
        print(f"✅ Found cached RNA embeddings at {rna_cache_path}. Loading...")
        adata = anndata.read_h5ad(rna_cache_path)
        # 【关键修改】提取 .X 并转为 Tensor
        z_rna = torch.from_numpy(adata.X)
    else:  
        print("\n=== RNA ===")
        adata = anndata.read_h5ad(RNA_FILE)
        if adata.obs_names.is_unique is False: adata.obs_names_make_unique()
        z_rna = cw_model.encode_rna(adata) # [N_cells, 512]
    
    # 3. Text Inference
    print("\n=== Text ===")
    with open(DISEASE_FILE, 'r') as f:
        omim_gene_sets = json.load(f)
    disease_names = list(omim_gene_sets.keys())
    
    # 构造更丰富的 Prompt 模拟训练数据 (Patient Summary)
    # 这一步对于 Zero-shot 效果至关重要
    prompts = []
    for d in disease_names:
        p = f"- Patient diagnosed with {d}." #epoch 1 2
        # p = f"{d}" # epoch3
        # p = f"The sample is diagnosed with {d}." #epoch 4
        # 使用和训练时类似的模板结构，激发模型的对齐能力
        # p = (
        #         f"**Patient Summary**  \n"
        #         f"- Patient diagnosed with {d}.  \n"
        #         f"\n"
        #         f"**Gene Interpretation**  \n"
        #         f"- No mutation data provided.  \n"
        #         f"\n"
        #         f"**Prognosis**  \n"
        #         f"- No prognosis information provided."
        #     )

        prompts.append(p)

    def compute_csls_sim(q_vecs, db_vecs, k=10):
        # ... (保持原样)
        sim_matrix = torch.matmul(q_vecs, db_vecs.T)
        topk_sim_q, _ = torch.topk(sim_matrix, k=k, dim=1)
        r_q = torch.mean(topk_sim_q, dim=1).unsqueeze(1)
        topk_sim_db, _ = torch.topk(sim_matrix, k=k, dim=0)
        r_db = torch.mean(topk_sim_db, dim=0).unsqueeze(0)
        return 2 * sim_matrix - r_q - r_db
        
    z_text = cw_model.encode_text(prompts) # [N_diseases, 512]
    
    # 4. Similarity Calculation
    print("\n=== Similarity ===")
    sim_matrix = compute_csls_sim(z_text.to(DEVICE), z_rna.to(DEVICE))
    sim_matrix = sim_matrix.cpu().numpy()

    # sim_matrix = torch.mm(z_text.to(DEVICE), z_rna.to(DEVICE).t()).cpu().numpy()
    
    df_sim = pd.DataFrame(sim_matrix, index=disease_names, columns=adata.obs_names)
    save_path = os.path.join(OUTPUT_DIR, f"disease_similarity_matrix_youtu_omics_frozen_{model_name}.csv")
    df_sim.to_csv(save_path)
    print(f"Saved to {save_path}")
    
    if not rna_cache_path.exists():
        # 5. Save Embeddings
        adata_rna_emb = anndata.AnnData(X=z_rna.numpy())
        adata_rna_emb.obs_names = adata.obs_names
        adata_rna_emb.write_h5ad(os.path.join(OUTPUT_DIR, f"rna_embeddings_youtu_omics_frozen_{model_name}.h5ad"))
    
    adata_text_emb = anndata.AnnData(X=z_text.numpy())
    adata_text_emb.obs_names = disease_names
    adata_text_emb.write_h5ad(os.path.join(OUTPUT_DIR, f"text_embeddings_youtu_omics_frozen_{model_name}.h5ad"))

if __name__ == "__main__":
    main()
