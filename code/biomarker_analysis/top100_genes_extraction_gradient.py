import copy
import json
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# Hotfix for transformers
import transformers
from captum.attr import LayerIntegratedGradients
from lifelines.utils import concordance_index
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

if not hasattr(transformers.utils, "LossKwargs"):
    class LossKwargs: pass
    transformers.utils.LossKwargs = LossKwargs

# 添加路径 (根据你的环境调整)
sys.path.append("../")
from omix.model.omni_fusion import \
    OmniFusionBlock
from omix.model.model_multi import PerformerModel
from omix.tokenizer import GeneVocab

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ====================================================================================
# 1. 配置参数 (必须与训练时一致)
# ====================================================================================

# 预训练模型路径 (用于读取配置文件 args.json 和 vocab)
# 请确保此路径存在，代码需要从中读取模型超参数
PRETRAIN_DIR = "../pretrain/save/omix_o_pretrain"

# Finetune 结果保存的路径 (即包含 fold1-fold5 的父目录)
FINETUNE_DIR = "survival_file_dir/survival_multimodal_finetune-performer-BRCA-gateFalse-hvgFalse-DROPOUT0.1"

# 模态配置
MODALITIES_ORDER = ['RNA', 'Protein', 'METHYL']
MODALITY_CHARS = {'RNA': 'R', 'Protein': 'P', 'METHYL': 'M'}
PAD_TOKEN = "<pad>"
PAD_VALUE = 51

# 显卡设置
gpu_index = 2
DEVICE = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(gpu_index)

BATCH_SIZE = 1  # 评估时可以使用与训练相同的 batch size
DROPOUT = 0.1 # 这里的Dropout主要影响模型构建结构，Eval模式下会被关闭

print(f"Evaluation Script")
print(f"Loading Config from: {PRETRAIN_DIR}")
print(f"Loading Finetuned Weights from: {FINETUNE_DIR}")
print(f"Device: {DEVICE}")

# ====================================================================================
# 2. 类定义 (直接复用训练代码)
# ====================================================================================

class OmniFusionBlockDataset(Dataset):
    """
    复用训练代码中的 Dataset
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
# New Class: Gene Importance Analyzer
# ====================================================================================

class GeneImportanceEvaluator:
    def __init__(self, model, device, vocab_dict):
        self.model = model
        self.device = device
        self.vocab = vocab_dict['RNA']
        
        # 词表映射修复
        if hasattr(self.vocab, 'get_stoi'):
            vocab_items = self.vocab.get_stoi().items()
        elif hasattr(self.vocab, 'stoi'):
            vocab_items = self.vocab.stoi.items()
        else:
            vocab_items = {} 
        self.inv_vocab = {v: k for k, v in vocab_items}

        # -------------------------------------------------------
        # Wrapper: 负责处理输入维度对齐 (解决 shape mismatch)
        # -------------------------------------------------------
        def model_forward_wrapper(rna_genes, rna_values, other_batch_data):
            # A. 维度对齐逻辑 (处理 Captum 的 n_steps 扩展)
            current_bs = rna_genes.shape[0]
            original_bs = other_batch_data['observed'].shape[0]
            
            batch_dict = {}
            if current_bs != original_bs:
                repeat_factor = current_bs // original_bs
                for k, v in other_batch_data.items():
                    if isinstance(v, torch.Tensor):
                        dims = [1] * v.dim(); dims[0] = repeat_factor
                        batch_dict[k] = v.repeat(*dims)
                    elif isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                        batch_dict[k] = tuple([t.repeat(*([1]*t.dim().__setitem__(0, repeat_factor) or [repeat_factor] + [1]*(t.dim()-1))) for t in v]) # 简写，逻辑同前
                        # 上面这行如果报错，用回之前的完整 for 循环写法即可
                        # 简单写法：
                        new_tuple = []
                        for t in v:
                            d = [1]*t.dim(); d[0] = repeat_factor
                            new_tuple.append(t.repeat(*d))
                        batch_dict[k] = tuple(new_tuple)
                    else:
                        batch_dict[k] = v
            else:
                batch_dict = copy.copy(other_batch_data)

            # B. 构造 RNA 输入
            pad_token_id = self.vocab[PAD_TOKEN]
            rna_mask = rna_genes.eq(pad_token_id)
            batch_dict['RNA'] = (rna_genes, rna_values, rna_mask)
            
            # C. 输出 Risk
            risk_score, _ = self.model(batch_dict)
            return risk_score

        self.forward_func = model_forward_wrapper
        
        # -------------------------------------------------------
        # Target: 直接锁定 RNA Encoder 模块
        # -------------------------------------------------------
        # self.target_layer = model.encoder_dict['RNA']
        self.target_layer = model.encoder_dict['RNA'].transformer_encoder.net
        self.lig = LayerIntegratedGradients(self.forward_func, self.target_layer)

    def compute_topk_genes(self, dataloader, top_k=20, num_steps=10):
        print(f"Computing Gene Importance...")
        self.model.eval() # Eval模式不影响求导，只要 requires_grad=True
        
        gene_attributions = {}
        gene_counts = {}
        pad_id = self.vocab[PAD_TOKEN]

        for batch in tqdm(dataloader, desc="Interpreting"):
            # 显存清理
            torch.cuda.empty_cache()
            
            try:
                # 准备数据
                batch_dict = {k: v if isinstance(v, list) else v.to(self.device) 
                              for k, v in batch.items() if k != 'RNA'}
                rna_genes = batch['RNA'][0].to(self.device)
                rna_values = batch['RNA'][1].to(self.device)
                
                # Baseline
                baseline_genes = torch.full_like(rna_genes, pad_id)
                baseline_values = torch.zeros_like(rna_values)
                
                # 计算归因
                # attribute_to_layer_input=False 表示我们要的是 Layer 的 Output
                attributions = self.lig.attribute(
                    inputs=(rna_genes, rna_values),
                    baselines=(baseline_genes, baseline_values),
                    additional_forward_args=(batch_dict,),
                    n_steps=num_steps,
                    internal_batch_size=BATCH_SIZE, # 关键：防爆显存
                    # attribute_to_layer_input=False,
                    attribute_to_layer_input=True
                )
                
                # -------------------------------------------------------
                # 关键修复：处理 Tuple 输出
                # -------------------------------------------------------
                # O-MIX Encoder 返回 (output_tensor, attn_weights)
                # Captum 对 Tuple 输出的层，会返回一个 Tuple 的 attributions
                if isinstance(attributions, tuple):
                    # print("Debug: Captum returned tuple, taking first element")
                    attributions = attributions[0] # 取 Hidden States 的梯度
                
                # 此时 attributions shape 应该是 [batch, seq_len, hidden_dim]
                # 验证一下是否有数值
                if attributions.abs().sum() == 0:
                    print('zeeeeeeeero....')
                    # 如果还是0，说明下游梯度没传过来，跳过本次循环避免干扰，但在 main 里必须解冻
                    continue

                # 聚合：在 Hidden Dim 维度求和 -> [batch, seq_len]
                # attributions: (batch, seq_len, hidden_dim)
                # token_importance: (batch, seq_len)

                token_importance = torch.norm(attributions, p=2, dim=-1)

                # # 归一化（可选）：让不同样本间的重要性有可比性
                # token_importance = token_importance / (token_importance.norm(p=2) + 1e-9)
                # token_importance = token_importance.detach().cpu().numpy()

                rna_genes_cpu = rna_genes.cpu().numpy()
                
                for b in range(token_importance.shape[0]):
                    for s in range(token_importance.shape[1]):
                        gene_id = int(rna_genes_cpu[b, s])
                        score = token_importance[b, s]
                        
                        if gene_id in self.inv_vocab:
                            gene_name = self.inv_vocab[gene_id]
                            # 过滤特殊 token
                            if gene_name not in [PAD_TOKEN, "<pad>", "<cls>", "<unk>", "UNK"]:
                                if gene_name not in gene_attributions:
                                    gene_attributions[gene_name] = 0.0
                                    gene_counts[gene_name] = 0
                                
                                # 这里已经是 Norm 了，本身就是非负的，直接累加
                                gene_attributions[gene_name] += score
                                gene_counts[gene_name] += 1

            except Exception as e:
                print(f"Error: {e}")
                continue

        # 统计排序
        avg_importance = []
        for gene, total in gene_attributions.items():
            if gene_counts[gene] > 0:
                avg_importance.append((gene, total / gene_counts[gene]))
        
        avg_importance.sort(key=lambda x: x[1], reverse=True)
        
        # 打印
        # 打印预览 (只打印前 print_top_k 个，防止刷屏)
        # print(f"\n[Preview] Top {print_top_k} Genes in this Fold:")
        # for g, s in avg_importance[:print_top_k]:
        #     print(f"{g:<15} | {s:.6f}")
            
        # 【关键修改】返回所有基因，不做切片
        return avg_importance

class SurvivalPredictor(nn.Module):
    """
    复用训练代码中的模型定义
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

        # Survival Head
        head_in_dim = embed_dim * len(MODALITIES_ORDER)
        self.head = nn.Sequential(
            nn.Linear(head_in_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1) 
        )

    def run_encoder(self, mod_name, batch_data, device):
        genes, vals, mask = batch_data
        genes, vals, mask = genes.to(device), vals.to(device), mask.to(device)
        enc_out = self.encoder_dict[mod_name](
            src=genes, values=vals, src_key_padding_mask=mask, return_transformer_output=True
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
        
        risk_score = self.head(final_repr).squeeze(-1)
        gate_loss = self.fusion_model.gate_loss()
        return risk_score, gate_loss

# ====================================================================================
# 3. 辅助函数
# ====================================================================================

def evaluate(model, dataloader, device):
    """
    推理函数
    """
    model.eval()
    all_risks = []
    all_times = []
    all_events = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            times = batch['time']
            events = batch['event']
            preds, _ = model(batch)
            
            all_risks.append(preds.cpu().numpy())
            all_times.append(times.cpu().numpy())
            all_events.append(events.cpu().numpy())
    
    all_risks = np.concatenate(all_risks)
    all_times = np.concatenate(all_times)
    all_events = np.concatenate(all_events)
    
    try:
        # C-Index (Lifelines expects: time, -risk, event for hazard models)
        c_idx = concordance_index(all_times, -all_risks, all_events)
    except Exception as e:
        print(f"C-Index Error: {e}")
        c_idx = 0.5
        
    return c_idx, all_times, all_events, all_risks

def get_modality_combinations_count():
    # 为了初始化 missing embeddings，我们需要知道组合数
    # 复用原来的逻辑
    from itertools import combinations
    ALL_CHARS = "".join(sorted([MODALITY_CHARS[m] for m in MODALITIES_ORDER]))
    chars = list(ALL_CHARS)
    all_combs = []
    for r in range(1, len(chars) + 1):
        all_combs.extend(combinations(chars, r))
    sorted_combs = [''.join(sorted(comb)) for comb in all_combs]
    full_modality_str = ''.join(sorted(chars))
    if full_modality_str in sorted_combs:
        sorted_combs.remove(full_modality_str)
        sorted_combs.insert(0, full_modality_str)
    return len(sorted_combs)

# ====================================================================================
# 4. Main Evaluation Loop
# ====================================================================================

def main():
    pretrain_path = Path(PRETRAIN_DIR)
    finetune_path = Path(FINETUNE_DIR)
    
    if not pretrain_path.exists():
        raise FileNotFoundError(f"Pretrain path not found: {pretrain_path}")
    if not finetune_path.exists():
        raise FileNotFoundError(f"Finetune path not found: {finetune_path}")

    # 1. 加载配置和词表 (为了构建模型结构)
    print("Loading arguments and vocab...")
    with open(pretrain_path / "args.json", "r") as f:
        model_configs = json.load(f)

    # 修复 vocab_mod
    vocab_mod = model_configs.get("modality_dict", {'RNA': 0, 'Protein': 1, 'METHYL': 2, PAD_TOKEN: 4})
    if PAD_TOKEN not in vocab_mod:
        vocab_mod[PAD_TOKEN] = len(vocab_mod)

    NUM_COMBINATIONS = get_modality_combinations_count()
    print(f"Number of modality combinations: {NUM_COMBINATIONS}")

    results_summary = []
    all_folds_preds = []
    all_folds_importance_df = []

    # 遍历 5 个 Fold
    for fold in range(1, 6):
        fold_dir = finetune_path / f"fold_{fold}"
        print(f"\n{'='*20} Processing Fold {fold} {'='*20}")
        
        # 检查必要文件
        files_needed = ["processed_data.pt", "best_model.pt", "test_idx.npy"]
        missing_files = [f for f in files_needed if not (fold_dir / f).exists()]
        if missing_files:
            print(f"Skipping Fold {fold}: Missing files {missing_files}")
            continue

        # ------------------------------------------------------------------
        # A. 加载数据
        # ------------------------------------------------------------------
        print(f"Loading data for Fold {fold}...")
        # processed_data.pt 包含了整个数据集 (train+val)，我们需要用 test_idx 来切分
        data_package = torch.load(fold_dir / "processed_data.pt", map_location='cpu')
        test_indices = np.load(fold_dir / "test_idx.npy")
        
        # 这里的 vocab_dict 也是从 processed_data 里拿，保证一致性
        vocab_dict = data_package['vocab_dict'] 

        # 重建 Metadata 字典以适应 Dataset __init__
        metadata_wrapper = {
            'observed_idx': data_package['observed_idx'],
            'modality_comb': data_package['modality_comb'],
            'events': data_package['events'],
            'times': data_package['times']
        }
        
        # 创建全量数据集
        full_dataset = OmniFusionBlockDataset(data_package['omics_data'], metadata_wrapper, vocab_dict)
        
        # 创建 Test Subset 和 DataLoader
        test_sub = torch.utils.data.Subset(full_dataset, test_indices)
        test_loader = DataLoader(test_sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        print(f"Test Set Size: {len(test_indices)}")

        # ------------------------------------------------------------------
        # B. 构建模型
        # ------------------------------------------------------------------
        print("Building model architecture...")
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

        model = SurvivalPredictor(
            encoder_dict, fusion_model, model_configs["layer_size"], vocab_mod, DROPOUT, NUM_COMBINATIONS
        ).to(DEVICE)

        # ------------------------------------------------------------------
        # C. 加载训练好的权重
        # ------------------------------------------------------------------
        model_path = fold_dir / "best_model.pt"
        print(f"Loading weights from {model_path}...")
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        # 你的保存代码是: {'model_state_dict': model.state_dict(), ...}
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint # 兼容直接保存 state_dict 的情况
            
        try:
            model.load_state_dict(state_dict, strict=True)
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Attempting with strict=False...")
            model.load_state_dict(state_dict, strict=False)

        print("Loading pretrained encoder weights...")
        pretrained_ckpt = torch.load(Path(PRETRAIN_DIR) / "model_e8.pt", map_location=DEVICE)
        # 假设 pretrained_ckpt['encoder_dict'] 结构是 {'RNA': state_dict, ...}
        # 如果使用了 nn.ModuleDict，加载方式可能略有不同，但因为你是普通字典，直接遍历加载：
        for mod in MODALITIES_ORDER:
            # 这一步把随机初始化的 encoder 变成了预训练状态
            model.encoder_dict[mod].load_state_dict(pretrained_ckpt['encoder_dict'][mod], strict=True)

        # if fold == 1: 
        print("\n=== Running Gradient Sanity Check ===")
        for param in model.fusion_model.parameters():
            param.requires_grad = True
            
        # 2. 解冻 Head (梯度出发点)
        for param in model.head.parameters():
            param.requires_grad = True
            
        # 3. 解冻 Encoder (梯度终点)
        # 虽然我们要的是 Encoder Output，但有些 PyTorch 版本如果 Module 本身
        # 被设为 frozen，钩子可能挂不上去，保险起见解冻。
        for param in model.encoder_dict['RNA'].parameters():
            param.requires_grad = True
        
        # 保持 eval 模式 (关闭 Dropout 的随机性)
        model.eval()
        # 1. 检查参数是否被冻结
        rna_encoder = model.encoder_dict['RNA']
        frozen_params = [p.requires_grad for p in rna_encoder.parameters()]
        if not any(frozen_params):
            print("CRITICAL WARNING: All parameters in RNA encoder are frozen (requires_grad=False)!")
            print("Fixing it now by enabling gradients...")
            for p in rna_encoder.parameters():
                p.requires_grad = True
        else:
            print(f"Parameters check: {sum(frozen_params)}/{len(frozen_params)} parameters have requires_grad=True. (Good)")

        # 2. 模拟一次反向传播，看梯度是否为0
        interpret_loader = DataLoader(test_sub, batch_size=1, shuffle=False)
        test_batch = next(iter(interpret_loader))
        test_batch_gpu = {k: v if isinstance(v, list) else v.to(DEVICE) for k, v in test_batch.items() if k != 'RNA'}
        # 构造输入
        rna_genes = test_batch['RNA'][0].to(DEVICE)
        rna_values = test_batch['RNA'][1].to(DEVICE)
        # 使用 Wrapper 逻辑
        pad_id = PAD_VALUE
        rna_mask = rna_genes.eq(pad_id)
        test_batch_gpu['RNA'] = (rna_genes, rna_values, rna_mask)
        
        # 前向
        model.zero_grad()
        risk, _ = model(test_batch_gpu)
        
        # 反向
        risk.mean().backward()
        
        # 检查 Encoder 第一层权重的梯度
        # 假设内部结构，你需要根据 print(rna_encoder) 调整
        # 试图获取一个线性层的梯度
        has_grad = False
        for name, param in rna_encoder.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                print(f"Gradient detected in: {name} (Sum: {param.grad.abs().sum().item():.6f})")
                has_grad = True
                break
        
        if not has_grad:
            print("CRITICAL: No gradients detected in RNA Encoder after backward pass!")
        else:
            print("Gradient flow confirmed. The model is fine.")

        print("=========================================\n")
        print(f"\nRunning Interpretation on Fold {fold} Test Set...")
        interpreter = GeneImportanceEvaluator(model, DEVICE, vocab_dict)
        # 使用 test_loader 的一部分数据来加速演示，或者使用全部
        fold_all_genes = interpreter.compute_topk_genes(test_loader, top_k=100)
        # 转 DataFrame
        fold_df = pd.DataFrame(fold_all_genes, columns=['Gene', 'Importance'])
        fold_df['Fold'] = fold
        
        # 保存该 Fold 的完整结果 (可能有上万行，但 csv 很小，不用担心)
        fold_df.to_csv(fold_dir / f"fold_{fold}_full_gene_importance.csv", index=False)
        
        # 加入总表
        all_folds_importance_df.append(fold_df)

        # ------------------------------------------------------------------
        # D. 推理与评估
        # ------------------------------------------------------------------
        c_idx, val_times, val_events, val_risks = evaluate(model, test_loader, DEVICE)
        print(f"Fold {fold} C-Index: {c_idx:.4f}")
        
        results_summary.append(c_idx)

        
        
        # 收集详细预测结果
        fold_df = pd.DataFrame({
            # 'Fold': [fold] * len(full_dataset),
            'Sample_Index': test_indices, # 原始数据集中的索引
            'Time': val_times,
            'Event': val_events,
            'Risk_Score': val_risks
        })
        all_folds_preds.append(fold_df)
        
        # 清理显存
        del model, encoder_dict, fusion_model, data_package, full_dataset
        torch.cuda.empty_cache()

    # ====================================================================================
    # 5. 汇总结果
    # ====================================================================================
    print(f"\n{'='*25} FINAL EVALUATION SUMMARY {'='*25}")
    if results_summary:
        mean_c = np.mean(results_summary)
        std_c = np.std(results_summary)
        print(f"Per Fold C-Index: {results_summary}")
        print(f"Mean C-Index: {mean_c:.4f} ± {std_c:.4f}")
    
    total_df = pd.concat(all_folds_importance_df, ignore_index=True)
        
    # 2. 按基因分组，计算 Mean, Std, Count
    # 这一步非常关键：它利用了5个Fold的所有数据
    agg_df = total_df.groupby('Gene')['Importance'].agg(['mean', 'std', 'count']).reset_index()
    
    # 3. 排序 (按 5 个 Fold 的平均重要性)
    agg_df = agg_df.sort_values(by='mean', ascending=False)
    
    # 4. 【最后一步】取 Top 100
    # 这里的 Top 100 是基于全局平均分选出来的，是最稳健的
    final_top100 = agg_df.head(100)
    
    print("\nFinal Top 10 Robust Biomarkers (Avg over 5 folds):")
    print(final_top100.head(10))
    
    # 5. 保存
    save_path = finetune_path / "FINAL_robust_top100_genes.csv"
    final_top100.to_csv(save_path, index=False)
    
    # 保存全量聚合表 (方便后续画图，比如画 Volcanic plot 或 Heatmap)
    agg_df.to_csv(finetune_path / "FINAL_all_genes_aggregated.csv", index=False)
    print(f"\nResults saved to {finetune_path}")


if __name__ == '__main__':
    main()
