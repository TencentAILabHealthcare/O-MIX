import copy
import math
from itertools import combinations
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn

from .moe_module import *
from .performer_pytorch import *


class OmniFusionBlock(nn.Module):
    def __init__(self, num_modalities, full_modality_index, total_length, hidden_dim, num_layers, num_experts, num_routers, top_k, num_heads=2, dropout=0.5, vocab_mod={'RNA':0, 'Protein': 1, 'METHYL':2, 'TEXT':3, '<pad>': 4}, device=None, n_cls=None, queue_K=1024):
        super(OmniFusionBlock, self).__init__()
        layers = []
        _sparse = True
        self.use_mod = True
        self.n_cls = n_cls
        if self.n_cls:
            # self.cls_decoder = ClsDecoder(num_modalities * hidden_dim, self.n_cls, nlayers=3)
            self.cls_decoder = FunnelDecoder(num_modalities * hidden_dim, self.n_cls, dropout)

        self.id2mod = {v: k for k, v in vocab_mod.items() if k not in ['<pad>', '<cls>', '<sep>']}

        self.adapters = nn.ModuleDict()
        for mod_name, mod_idx in vocab_mod.items():
            if mod_name in ['<pad>', '<cls>', '<sep>']: # 跳过特殊 token
                continue
            # 为每个模态创建一个 Adapter
            self.adapters[mod_name] = ModalityAdapter(hidden_dim, hidden_dim, dropout=dropout)
        
        if self.use_mod:
            self.mod_encoder = ModalityTokenEncoder(len(vocab_mod), hidden_dim, padding_idx=vocab_mod['<pad>'])
            # hidden_dim = hidden_dim * 2

        layers.append(TransformerEncoderLayer(num_experts, num_routers, hidden_dim, num_head=num_heads, dropout=dropout, hidden_times=2, mlp_sparse=_sparse, full_modality_index=full_modality_index, top_k=top_k))
        for _ in range(num_layers - 1):
            _sparse = not _sparse
            layers.append(TransformerEncoderLayer(num_experts, num_routers, hidden_dim, num_head=num_heads, dropout=dropout, hidden_times=2, mlp_sparse=_sparse, full_modality_index=full_modality_index, top_k=top_k))
        # layers.append(MLP(hidden_dim*num_modalities, hidden_dim, output_dim, num_layers_pred, activation=nn.ReLU(), dropout=0.5))
        
        self.network = nn.Sequential(*layers)
        self.combination_to_index = self._create_combination_index(num_modalities)
        # self.simsiam = BimodalSimSiam(hidden_dim)
        # self.simclr = BimodalSimCLR(hidden_dim)
        # self.simclr = MultiModalSimCLR(hidden_dim, device=device)

        modality_names = [k for k, v in sorted(vocab_mod.items(), key=lambda x: x[1]) if k not in ['<pad>']]
        self.simclr = MultiModalSimCLR(
            dim=hidden_dim, 
            modality_names=modality_names,
            K=queue_K,
            T=0.07, 
            device=device
        )
        self.adaln = MultiModalAdaLN(hidden_dim, num_modalities=num_modalities)

    def forward(self, *inputs, masks=None, expert_indices=None, mod_types=None, return_cell_embedding=False, return_cls_result=False,
                # 【新增参数】用于内部计算 SimCLR
                calculate_simclr=False,
                batch_observed=None,
                pre_fusion_map=None,    # Query Features
                pre_fusion_map_k=None,  # Key Features (from Momentum Encoder)
                modality_pairs=None,
                text_loss_weight=None,
                bio_loss_weight=None,
                training=True,
                protein_loss_weight=None):
        self.training = training
        # 拿到每个modality的长度
        chunk_size = [input.shape[1] for input in inputs]
        x = torch.cat(inputs, dim=1)

        if self.use_mod:
            mod_emb = self.mod_encoder(mod_types)
            x = x + mod_emb

        x = torch.split(x, chunk_size, dim=1)

        for i in range(len(self.network)):
            if expert_indices is not None and hasattr(self.network[i], 'set_expert_index'):
                self.network[i].set_expert_index(expert_indices)
            x = self.network[i](x, attn_mask=masks)
        # return x
        fusion_output = x

        if return_cell_embedding:
            sample_embedding = torch.cat(fusion_output, dim=-1)
            return torch.squeeze(sample_embedding)

        if return_cls_result:
            sample_embedding = torch.cat(fusion_output, dim=-1)
            sample_embedding = torch.squeeze(sample_embedding)
            return self.cls_decoder(sample_embedding)

        # ================== 2. 【新增】SimCLR 内部计算逻辑 ==================
        simclr_loss_val = torch.tensor(0.0, device=x[0].device)

        sim_stats_agg = {'pos_sum': 0.0, 'neg_sum': 0.0, 'pair_count': 0}

        if calculate_simclr and batch_observed is not None and pre_fusion_map is not None and modality_pairs is not None and pre_fusion_map_k is not None:
            sim_losses = []
            if self.training:
                self.simclr.update_momentum_projectors()

            for mod_idx1, mod_idx2 in modality_pairs:
                mod_name1 = self.id2mod.get(mod_idx1)
                mod_name2 = self.id2mod.get(mod_idx2)
                if mod_name1 is None or mod_name2 is None: continue

                # ==============================================================================
                # 【修复点】检查当前 Batch 是否真的包含这两个模态的数据
                # 如果某个模态在这个 Batch 完全缺失，它就不会出现在 pre_fusion_map 或 pre_fusion_map_k 中
                # 直接跳过，防止 KeyError
                # ==============================================================================
                if mod_idx1 not in pre_fusion_map or mod_idx2 not in pre_fusion_map:
                    continue
                if mod_idx1 not in pre_fusion_map_k or mod_idx2 not in pre_fusion_map_k:
                    continue
                # ==============================================================================

                mask1 = batch_observed[:, mod_idx1]
                mask2 = batch_observed[:, mod_idx2]
                pair_present_mask = mask1 & mask2
                
                x1_q = pre_fusion_map[mod_idx1][:, 0, :]
                x2_k = pre_fusion_map_k[mod_idx2][:, 0, :]
                
                x2_q = pre_fusion_map[mod_idx2][:, 0, :]
                x1_k = pre_fusion_map_k[mod_idx1][:, 0, :]
                
                loss_1to2, stats1 = self.simclr(x1_q, mod_name1, x2_k, mod_name2, 
                                                mask=pair_present_mask, update_queue=True,training=self.training)
                
                # 2. Mod2 检索 Mod1 (Query: Mod2, Key: Mod1 Queue)
                # update_queue=True 表示把 x1_k 加入 Queue_Mod1
                loss_2to1, stats2 = self.simclr(x2_q, mod_name2, x1_k, mod_name1, 
                                                mask=pair_present_mask, update_queue=True,training=self.training)

                pair_loss = (loss_1to2 + loss_2to1) / 2

                if 'TEXT' in [mod_name1, mod_name2] and text_loss_weight:
                    pair_loss = pair_loss * text_loss_weight
                # 文本不在，全是bio2bio的时候，加上权重
                if 'TEXT' not in [mod_name1, mod_name2] and bio_loss_weight:
                    pair_loss = pair_loss * bio_loss_weight
                if 'Protein' in [mod_name1, mod_name2] and protein_loss_weight:
                    pair_loss = pair_loss * protein_loss_weight

                sim_losses.append(pair_loss)

                if stats1['count'] > 0:
                    sim_stats_agg['pos_sum'] += (stats1['pos_sim'] + stats2['pos_sim']) / 2
                    sim_stats_agg['neg_sum'] += (stats1['neg_sim'] + stats2['neg_sim']) / 2
                    sim_stats_agg['pair_count'] += 1

            if sim_losses:
                simclr_loss_val = torch.stack(sim_losses).mean()

        final_sim_stats = {}
        if sim_stats_agg['pair_count'] > 0:
            final_sim_stats['avg_pos_sim'] = sim_stats_agg['pos_sum'] / sim_stats_agg['pair_count']
            final_sim_stats['avg_neg_sim'] = sim_stats_agg['neg_sum'] / sim_stats_agg['pair_count']

        # --- [修改点 5：返回三个值] ---
        return fusion_output, simclr_loss_val, final_sim_stats
    
    def run_adapter(self, modality_name, x):
        if modality_name in self.adapters:
            return self.adapters[modality_name](x)
        return x

    def gate_loss(self):
        g_loss = []
        for mn, mm in self.named_modules():
            if hasattr(mm, 'all_gates'):
                for i in range(len(mm.all_gates)):
                    i_loss = mm.all_gates[f'{i}'].get_loss_clear(clear=True)  # 清理loss
                    if i_loss is not None:
                        g_loss.append(i_loss)
                    mm.all_gates[f'{i}'].get_topk_logit(clear=True)
        return sum(g_loss) if g_loss else torch.tensor(0.0, device=next(self.parameters()).device)

    def get_simsiam(self, x1, x2):
        return self.simsiam(x1, x2)

    def get_simclr(self, x1, mod_name1, x2, mod_name2, training):
        # 1. 直接调用 self.simclr (触发 forward)
        # 2. 关键：设置 update_queue=False。
        #    train 循环中调用此方法是为了计算 dummy loss，不能让 dummy 数据污染 Memory Bank (Queue)
        # 3. MoCo 返回的是 (loss, stats)，这里只需要 loss
        loss, _ = self.simclr(x1, mod_name1, x2, mod_name2, update_queue=False, training=training)
        return loss

    def get_z(self, x, mod_name):
        return self.simclr.project(x, mod_name)

    # def get_z(self, x, mod1_name, mod2_name):
        # return self.simclr.project(x, mod1_name, mod2_name)

    def get_adaln(self, x, list_of_cls_tokens):
        return self.adaln(x, list_of_cls_tokens)

    def _create_combination_index(self, num_modalities):
        combinations_list = []
        for r in range(1, num_modalities + 1):
            combinations_list.extend(combinations(range(num_modalities), r))
        combination_to_index = {tuple(sorted(comb)): idx for idx, comb in enumerate(combinations_list)}
        return combination_to_index

    def assign_expert(self, combination):
        index = self.combination_to_index.get(tuple(sorted(combination)))
        return index

    def set_full_modality(self, is_full_modality):
        for layer in self.network:
            if hasattr(layer, 'set_full_modality'):
                layer.set_full_modality(is_full_modality)

class MultiModalAdaLN(nn.Module):
    def __init__(self, d_model, num_modalities=4):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        input_dim = num_modalities * d_model
        
        self.proj = nn.Linear(input_dim, 2 * d_model) # 输出还是gamma和beta
        
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, condition):
        style = self.proj(condition)
        gamma, beta = style.view(style.size(0), 1, -1).chunk(2, dim=-1)
        
        # 3. FiLM 调制
        return self.norm(x) * (1 + gamma) + beta

class ModalityTokenEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, embsize)
        x = self.enc_norm(x)
        return x


class ModalityAdapter(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
    
    def forward(self, x):
        return self.net(x)
    

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation=nn.ReLU(), dropout=0.5):
        super(MLP, self).__init__()
        layers = []
        self.drop = nn.Dropout(dropout)
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation)
            layers.append(self.drop)
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation)
                layers.append(self.drop)
            layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    """
    # 如果没有初始化分布式训练，直接返回
    if not dist.is_initialized():
        return tensor

    # 1. 准备容器
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    
    # 2. 执行收集
    dist.all_gather(tensors_gather, tensor, async_op=False)

    # 3. 恢复梯度（虽然这里是no_grad，但保留此逻辑是MoCo官方标准）
    tensors_gather[dist.get_rank()] = tensor
    
    # 4. 拼接
    return torch.cat(tensors_gather, dim=0)

@torch.no_grad()
def momentum_update(model_q, model_k, m=0.999):
    """
    Momentum update of the key encoder
    model_k = m * model_k + (1 - m) * model_q
    """
    for param_q, param_k in zip(model_q.parameters(), model_k.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1. - m)


class MultiModalSimCLR(nn.Module):
    def __init__(self, dim, K=65536, m=0.999, T=0.07, modality_names=['RNA', 'Protein', 'METHYL', 'TEXT'], device=None):
        """
        dim: feature dimension (output of projector)
        K: queue size; number of negative keys (default: 65536)
        m: momentum coefficient (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MultiModalSimCLR, self).__init__()
        
        self.K = K
        self.m = m
        self.T = T
        # self.T = nn.Parameter(torch.ones([]) * 0.07)
        self.modality_names = modality_names
        self.device = device

        # 1. Online Projectors (Query) & Momentum Projectors (Key)
        # 改为共享空间投影：Projector(Modality) -> Shared Space
        self.projectors_q = nn.ModuleDict()
        self.projectors_k = nn.ModuleDict()

        for mod in modality_names:
            # 定义 Projector 结构
            proj = nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                # nn.Linear(dim, dim)
                nn.Linear(dim, dim, bias=False)
            )
            self.projectors_q[mod] = proj
            
            # Key Projector 必须与 Query 结构一致
            self.projectors_k[mod] = copy.deepcopy(proj)
            
            # Key Projector 不需要梯度
            for param in self.projectors_k[mod].parameters():
                param.requires_grad = False

        # 2. 创建队列 (Queue)
        # 为每个模态维护一个队列，存储的是经过 Projector 映射后的特征
        for mod in modality_names:
            self.register_buffer(f"queue_{mod}", torch.randn(dim, K))
            self.register_buffer(f"queue_ptr_{mod}", torch.zeros(1, dtype=torch.long))
            self.queue_norm(mod) # 初始化归一化

    def queue_norm(self, mod_name):
        q_name = f"queue_{mod_name}"
        queue = getattr(self, q_name)
        setattr(self, q_name, F.normalize(queue, dim=0))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, mod_name, mask=None):
        """
        keys: (Batch_Size, Dim) 当前卡的数据，包含全0向量
        mask: (Batch_Size, ) True表示有效数据
        """
        # 1. 收集所有显卡的数据 -> (World_Size * B, Dim)
        all_keys = concat_all_gather(keys)

        # 2. 关键修改：利用 Mask 剔除全 0 向量
        if mask is not None:
            # 必须先转 float 才能 all_gather
            all_masks = concat_all_gather(mask.float())
            all_masks = all_masks.bool() # 转回 bool
            
            # 全局过滤：只保留有效的 Key
            valid_keys = all_keys[all_masks]
        else:
            valid_keys = all_keys

        # 如果运气极差，所有卡全是 0 向量（几乎不可能），直接返回
        if valid_keys.shape[0] == 0:
            return

        batch_size = valid_keys.shape[0]
        q_name = f"queue_{mod_name}"
        ptr_name = f"queue_ptr_{mod_name}"

        queue = getattr(self, q_name)
        ptr = int(getattr(self, ptr_name))

        # 3. 入队 (使用过滤后的 clean data)
        if ptr + batch_size <= self.K:
            queue[:, ptr:ptr + batch_size] = valid_keys.T
            ptr = (ptr + batch_size) % self.K
        else:
            rem = self.K - ptr
            queue[:, ptr:self.K] = valid_keys.T[:, :rem]
            queue[:, 0:batch_size - rem] = valid_keys.T[:, rem:]
            ptr = batch_size - rem

        setattr(self, ptr_name, torch.tensor([ptr], dtype=torch.long, device=keys.device))

    @torch.no_grad()
    def update_momentum_projectors(self):
        """在 Train Loop 外部调用，更新 Projector 参数"""
        for mod in self.modality_names:
            momentum_update(self.projectors_q[mod], self.projectors_k[mod], self.m)

    def project_q(self, x, mod_name):
        return F.normalize(self.projectors_q[mod_name](x), dim=-1)

    @torch.no_grad()
    def project_k(self, x, mod_name):
        return F.normalize(self.projectors_k[mod_name](x), dim=-1)
        
    # 为了兼容你原来的 Top-k 检索代码，保留 project 接口，指向 project_q
    def project(self, x, src_mod, tgt_mod=None):
        # tgt_mod 参数不再需要，因为映射到了共享空间，但为了接口兼容保留
        if src_mod not in self.projectors_q:
             return F.normalize(x, dim=-1)
        return self.project_q(x, src_mod)

    def forward(self, x_q, mod_q, x_k, mod_k, mask=None, update_queue=True, training=True):
        self.training = training
        
        # 1. 计算 Loss (保持原样)
        q = self.project_q(x_q, mod_q) 
        k = self.project_k(x_k, mod_k) 

        # 计算相似度
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        queue_k = getattr(self, f"queue_{mod_k}").clone().detach()
        l_neg = torch.einsum('nc,ck->nk', [q, queue_k])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels, reduction='none')
        
        stats = {
            'pos_sim': l_pos.mean().item(),
            'neg_sim': l_neg.mean().item(),
            'count': x_q.shape[0],
            'temp': self.T
        }
        
        # Loss Masking (当前 Batch 计算 Loss 时忽略 0 向量)
        if mask is not None:
            if mask.sum() > 0:
                loss = (loss * mask).sum() / mask.sum()
            else:
                loss = loss.sum() * 0.0
        else:
            loss = loss.mean()

        # 2. 更新队列 (关键修改：传入 mask)
        if self.training and update_queue:
            # 这里传入 mask，_dequeue_and_enqueue 内部会负责过滤
            self._dequeue_and_enqueue(k, mod_k, mask=mask)

        return loss, stats


class L2Norm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.normalize(x, dim=-1)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, 
                num_experts,
                num_routers,
                d_model, 
                num_head, 
                dropout=0.1, 
                activation=nn.GELU, 
                hidden_times=2, 
                mlp_sparse = False, 
                self_attn = True,
                full_modality_index=4,
                top_k=2,
                world_size=1,
                **kwargs) -> None:
        super(TransformerEncoderLayer, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation()

        self.attn = SelfAttention(
            d_model,
            causal = False, 
            heads = num_head, 
            dim_head = d_model // num_head, 
            dropout = dropout, 
        )
        # self.attn = Attention(
        #     d_model, num_heads=num_head, qkv_bias=False, attn_drop=dropout, proj_drop=dropout)
        
        self.mlp_sparse = mlp_sparse
        self.self_attn = self_attn
        self.expert_index = None
        self.full_modality_index = full_modality_index

        if self.mlp_sparse:
            self.mlp = FMoETransformerMLP(num_expert=num_experts, n_router=num_routers, d_model=d_model, d_hidden=d_model * hidden_times, activation=nn.GELU(), top_k=top_k, world_size=world_size, **kwargs)
        else:
            self.mlp = MLP(input_dim=d_model, hidden_dim=d_model * hidden_times, output_dim=d_model, num_layers=2, activation=nn.GELU(), dropout=dropout)

    def forward(self, x, attn_mask = None):
        if self.self_attn:
            # 拿到每个modality长度， 如果三个modality，那就是3个items
            chunk_size = [item.shape[1] for item in x]
            x_concat = torch.cat(x, dim=1)  # 保存原始输入
            x_norm = self.norm1(x_concat)
            kv = x_norm

            if attn_mask is not None:
                key_padding_mask = torch.cat(attn_mask, dim=1)
                performer_mask = ~key_padding_mask
            else:
                performer_mask = attn_mask

            attn_out = self.attn(x_norm, mask=performer_mask)

            # attn_mask = torch.cat(attn_mask, dim=1)
            # attn_out = self.attn(x_norm, kv, attn_mask)

            x = x_concat + self.dropout1(attn_out)  # 正确的残差连接
            x = torch.split(x, chunk_size, dim=1)
            x = [item for item in x]
            if self.mlp_sparse:
                for i in range(len(chunk_size)):
                    # 分别对每个modality进行mlp前向传播
                    x[i] = x[i] + self.dropout2(self.mlp(self.norm2(x[i]), self.expert_index))
            else:
                for i in range(len(chunk_size)):
                    x[i] = x[i] + self.dropout2(self.mlp(self.norm2(x[i])))
        else:
            chunk_size = [item.shape[1] for item in x]
            x = [item for item in x]
            for i in range(len(chunk_size)):
                other_m = [x[j] for j in range(len(chunk_size)) if j != i]
                other_m = torch.cat([x[i], *other_m], dim=1)
                attn_mask = torch.cat(attn_mask, dim=1)
                x[i] = self.attn(x[i], other_m, attn_mask)
            x = [x[i]+self.dropout1(x[i]) for i in range(len(chunk_size))]
            if self.mlp_sparse:
                for i in range(len(chunk_size)):
                    x[i] = x[i] + self.dropout2(self.mlp(self.norm2(x[i]), self.expert_index))
            else:
                for i in range(len(chunk_size)):
                    x[i] = x[i] + self.dropout2(self.mlp(self.norm2(x[i])))
        return x

    def set_expert_index(self, expert_index):
        self.expert_index = expert_index

    def set_full_modality(self, is_full_modality):
        if hasattr(self.mlp, 'set_full_modality'):
            self.mlp.set_full_modality(is_full_modality)


class ExprDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
    ):
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        self.fc = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, 1),
        )
        self.explicit_zero_prob = explicit_zero_prob
        if explicit_zero_prob:
            self.zero_logit = nn.Sequential(
                nn.Linear(d_in, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, 1),
            )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        pred_value = self.fc(x).squeeze(-1)  # (batch, seq_len)

        if not self.explicit_zero_prob:
            return dict(pred=pred_value)
        zero_logits = self.zero_logit(x).squeeze(-1)  # (batch, seq_len)
        zero_probs = torch.sigmoid(zero_logits)
        return dict(pred=pred_value, zero_probs=zero_probs)
        # TODO: note that the return currently is only for training. Since decoder
        # is not used in the test setting for the integration task, the eval/inference
        # logic is not implemented yet. However, remember to implement it when
        # the decoder is used in any test setting. The inference logic will need
        # to sample from the bernoulli distribution with the zero_probs.

class TextEncoder(nn.Module):
    """
    一个简单的编码器，用于处理已经存在的文本嵌入。
    它将输入的高维嵌入投影到模型所需的维度。
    """
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
        
        # 为了与PerformerModel的接口保持一致，我们定义一个空的解码器
        # 因为文本模态没有重建任务
        self.decoder = nn.Identity()
        self.mvc_decoder = nn.Identity()

    def forward(self, embeddings, attention_mask=None, **kwargs):
        """
        前向传播函数。
        
        Args:
            embeddings (torch.Tensor): 形状为 (batch_size, seq_len, input_dim) 的嵌入向量。
            attention_mask (torch.Tensor): 形状为 (batch_size, seq_len) 的注意力掩码。
        
        Returns:
            元组: (处理后的嵌入, None) 以匹配 PerformerModel 的输出签名。
        """
        target_dtype = self.projection[0].weight.dtype
        
        # 将输入 embeddings 转换为与模型权重一致的类型
        projected_embeddings = self.projection(embeddings.to(target_dtype))
        
        return projected_embeddings, None

    def decode_only(self, hidden_x, gene_embs):
        """
        解码函数，在文本模态中不执行任何操作。
        返回一个包含零张量的字典，以防止在训练循环中出错。
        """
        return {"mlm_output": torch.tensor(0.0, device=hidden_x.device), 
                "mvc_output": torch.tensor(0.0, device=hidden_x.device)}

from transformers import AutoModel, AutoTokenizer


class TrainableTextEncoder(nn.Module):
    def __init__(self, model_name_or_path, output_dim, dropout=0.1, trust_remote_code=True):
        super().__init__()
        # 加载预训练模型 (Youtu)
        self.bert = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        
        # 冻结 BERT 主体参数 (稍后由 LoRA 接管，这里先冻结是个好习惯，虽然后面会被LoRA逻辑覆盖)
        for param in self.bert.parameters():
            param.requires_grad = False
            
        self.hidden_size = self.bert.config.hidden_size
        
        # 投影层：把 BERT 维度 (e.g. 768/1024) 映射到 O-MIX 的 layer_size (e.g. 512)
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )

    def mean_pooling(self, hidden_state, attention_mask):
        # 你的第一套代码中的 pooling 逻辑
        s = torch.sum(hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
        d = attention_mask.sum(dim=1, keepdim=True).float()
        embedding = s / d
        return embedding

    def forward(self, input_ids, attention_mask):
        # BERT Forward
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        
        # Pooling
        embeddings = self.mean_pooling(last_hidden_state, attention_mask)
        
        # Normalize (第一套代码逻辑)
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        
        # Projection -> (Batch, output_dim)
        # 此时维度是 (B, Dim)，为了适配 O-MIX 融合层，可能需要 unsqueeze 变成 (B, 1, Dim)
        projected = self.projection(embeddings)
        return projected.unsqueeze(1), None # 返回 (B, 1, Dim), None(模拟gene_token_embs)

class FunnelDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,  # 这里传入 1536 (512*3)
        output_dim: int = 1,
        dropout: float = 0.3 # 加上 Dropout 防止过拟合
    ):
        super().__init__()
        
        # 第一层：大幅压缩 (1536 -> 512)
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 128),
            # nn.BatchNorm1d(512),  # BN 对回归任务收敛很有帮助
            nn.GELU(),            # GELU 比 ReLU 更平滑
            nn.Dropout(dropout)
        )
        
        # 第二层：进一步压缩 (512 -> 128)
        self.layer2 = nn.Sequential(
            nn.Linear(128, 32),
            # nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 输出层 (64 -> 1)
        self.head = nn.Linear(32, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.head(x)
