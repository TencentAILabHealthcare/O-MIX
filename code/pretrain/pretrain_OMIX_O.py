import copy
import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path

import anndata
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from anndata import AnnData
from peft import LoraConfig, get_peft_model
from scipy.sparse import issparse
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchtext._torchtext import Vocab as VocabPybind
from torchtext.vocab import Vocab

sys.path.insert(0, "../")
import omix as omix_logger
from omix.loss import (criterion_neg_log_bernoulli, masked_mse_loss,
                        masked_relative_error)
from omix.model.omni_fusion import (
    OmniFusionBlock, TextEncoder, TrainableTextEncoder)
from omix.model.model_multi import PerformerModel
from omix.tokenizer.gene_tokenizer import GeneVocab
from omix.trainer_omix_o import (
    DataCollator, MultiModalDataset, create_flexmoe_input,
    define_wandb_metrcis, evaluate, train)
from omix.utils import category_str2int, eval_scib_metrics, set_seed

sc.set_figure_params(figsize=(4, 4))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

hyperparameter_defaults = dict(
    task = 'multiomic',
    seed=42,
    dataset_name="omix_o_pretrain", # Dataset name
    do_train=True, # Flag to indicate whether to do update model parameters during training
    load_model_RNA="save/rna_pretrain",
    load_model_Protein="save/protein_pretrain",
    load_model_methyl="save/methylation_pretrain",
    GEP=True, # Gene expression modelling
    GEPC=True, # Gene expression modelling for cell objective
    CLS=False,
    ESC=False,
    DAR = False, # DAR objective weight for batch correction
    DSBN = False,  # Domain-spec batchnorm,
    mask_ratio=0.15, # Default mask ratio
    explicit_zero_prob = False,  # whether explicit bernoulli for zeros
    ecs_thres=0,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    dab_weight=0.0,
    use_batch_labels = False,
    use_mod = False,
    per_seq_batch_sample = False,
    epochs=10, # Default number of epochs for fine-tuning
    input_layer_key = "X_binned", # Default expression value binning in data pre-processing
    n_bins=51, # Default number of bins for value binning in data pre-processing
    max_seq_len_protein = 17007+1,
    max_seq_len_rna = 19205+1,
    max_seq_len_methyl = 12920+1,
    max_seq_len_text=1024,
    lr=1e-4, # Default learning rate for fine-tuning
    batch_size=32, # Default batch size for fine-tuning
    layer_size=512,
    nlayers=6,
    nhead=8, # if load model, batch_size, layer_size, nlayers, nhead will be ignored
    dropout=0.2, # Default dropout rate during model fine-tuning
    schedule_ratio=0.95,  # Default rate for learning rate decay
    save_eval_interval=1, # Default model evaluation interval
    log_interval=20, # Default log interval
    fast_transformer=False, # Default setting
    pre_norm=False, # Default setting
    amp=False,  # Default setting: Automatic Mixed Precision
    pad_token = "<pad>",
    mask_value = 52,
    pad_value = 51,
    include_zero_gene = False,
    num_layers_fus=3,
    num_experts=16,
    num_routers=1,
    top_k=4,
    modality_dict={'RNA':0, 'Protein': 1, 'METHYL':2}, # 删掉 TEXT
    num_modalities=3, # 改为 3
    uniformity_weight=1,
    gate_loss_weight=1,
    simsiam_loss_weight=10,
    text_loss_weight=0,
    bio_loss_weight=1,
    protein_loss_weight=5,
    ranking_epoch=0,
    gradient_accumulation_steps=2,
    frozen_encoder=True,
    gradient_checkpointing=False
)



import datetime

# --- 分布式训练设置 ---
dist.init_process_group(backend="nccl",timeout=datetime.timedelta(minutes=120)) # 使用 NCCL 后端，GPU通信效率高
local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)
print(f"[init] "
      f"rank: {rank}, "
      f"world_size: {world_size}, "
      f"local_rank: {local_rank}, "
      f"device: {device}")

# rank = 0
# device = torch.device("cuda:0")

if rank == 0:
    # 从环境变量读取 WandB key，避免硬编码泄漏。
    # 用法：export WANDB_API_KEY=<your_key>   或   wandb login
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)

    run = wandb.init(
        config=hyperparameter_defaults,
        project="O-MIX",
        reinit=True,
        settings=wandb.Settings(start_method="fork"),
    )
    config = wandb.config
    print(config)
else:
    from wandb.sdk.wandb_config import Config
    config = Config()
    config.update(hyperparameter_defaults)

set_seed(config.seed)
# settings for input and preprocessing
special_tokens = [config.pad_token, "<cls>", "<eoc>"]
dataset_name = config.dataset_name

if rank == 0:
    save_dir = Path(f"./save/dev_{config.dataset_name}-DDP-{time.strftime('%b%d-%H-%M')}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = omix_logger.logger
    omix_logger.utils.add_file_handler(logger, save_dir / "run.log")
    logger.info(f"Save directory: {save_dir}")
    logger.info(f"save to {save_dir}")
    logger.info(config)
    logger.info(config.modality_dict.keys())
else:
    logger = None # 其他进程不需要 logger

youtu_model_path = "./omix/Youtu_embedding" 

print('start loading')
RNA_file = '../../data/pretraining_data/rna_full_data.h5ad'
Protein_file = '../../data/pretraining_data/protein_all_data.h5ad'
methyl_file = '../../data/pretraining_data/methylation_full_data.h5ad'


adata = anndata.read_h5ad(RNA_file, backed='r')
adata_protein = anndata.read_h5ad(Protein_file, backed='r')
adata_methyl = anndata.read_h5ad(methyl_file, backed='r')

adata.obs.index = adata.obs_names
adata_protein.obs.index = adata_protein.obs_names
adata_methyl.obs.index = adata_methyl.obs_names

adata.var.index = adata.var_names
adata_protein.var.index = adata_protein.var_names
adata_methyl.var.index = adata_methyl.var_names

all_ptid = sorted(list(set(adata.obs.index.tolist() + 
                           adata_protein.obs.index.tolist() + 
                           adata_methyl.obs.index.tolist())))

adata.var["gene_name"] = adata.var.index.tolist()
adata_protein.var["protein_name"] = adata_protein.var.index.tolist()
adata_methyl.var["methyl_name"] = adata_methyl.var.index.tolist()
# print('loading finish')
data_is_raw = False

vocab_dict = {}
if config.load_model_RNA is not None:
    model_dir = Path(config.load_model_RNA)
    model_config_file_RNA = model_dir / "args.json"
    model_RNA_file = model_dir / "model_e15.pt"
    vocab_file = model_dir / "vocab.json"

    vocab_RNA = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab_RNA:
            vocab_RNA.append_token(s)

    vocab_dict['RNA'] = vocab_RNA
    
    # 手动计算 gene_ids，如果基因不在 vocab 里，就赋值为 pad_token 的 ID
    # 这样在 dataset 里 tokenize 时，这些基因会被忽略
    pad_id = vocab_RNA[config.pad_token]
    gene_ids_list = []
    for gene in adata.var_names.tolist():
        if gene in vocab_RNA:
            gene_ids_list.append(vocab_RNA[gene])
        else:
            gene_ids_list.append(pad_id)
    
    gene_ids_RNA = np.array(gene_ids_list, dtype=int)
    
    if rank == 0:
        # 统计一下有多少有效基因
        valid_count = np.sum(gene_ids_RNA != pad_id)
        logger.info(
            f"match {valid_count}/{len(gene_ids_RNA)} genes "
            f"in vocabulary of size {len(vocab_RNA)}."
        )

if config.load_model_Protein is not None:
    model_dir = Path(config.load_model_Protein)
    model_config_file_Protein = model_dir / "args.json"
    model_Protein_file = model_dir / "model_e5.pt"
    vocab_file = model_dir / "vocab.json"

    vocab_Protein = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab_Protein:
            vocab_Protein.append_token(s)

    vocab_dict['Protein'] = vocab_Protein
    
    pad_id = vocab_Protein[config.pad_token]
    gene_ids_list = []
    for gene in adata_protein.var_names.tolist():
        if gene in vocab_Protein:
            gene_ids_list.append(vocab_Protein[gene])
        else:
            gene_ids_list.append(pad_id)
    
    proteins_ids_Protein = np.array(gene_ids_list, dtype=int)
    
    if rank == 0:
        valid_count = np.sum(proteins_ids_Protein != pad_id)
        logger.info(
            f"match {valid_count}/{len(proteins_ids_Protein)} protein "
            f"in vocabulary of size {len(vocab_Protein)}."
        )

if config.load_model_methyl is not None:
    model_dir = Path(config.load_model_methyl)
    model_config_file_methyl = model_dir / "args.json"
    model_methyl_file = model_dir / "model_e40.pt"
    vocab_file = model_dir / "vocab.json"

    vocab_methyl = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab_methyl:
            vocab_methyl.append_token(s)

    vocab_dict['METHYL'] = vocab_methyl
    
    pad_id = vocab_methyl[config.pad_token]
    gene_ids_list = []
    for gene in adata_methyl.var_names.tolist():
        if gene in vocab_methyl:
            gene_ids_list.append(vocab_methyl[gene])
        else:
            gene_ids_list.append(pad_id)
            
    gene_ids_METHYL = np.array(gene_ids_list, dtype=int)

    if rank == 0:
        valid_count = np.sum(gene_ids_METHYL != pad_id)
        logger.info(
            f"match {valid_count}/{len(gene_ids_METHYL)} methylation "
            f"in vocabulary of size {len(vocab_methyl)}."
        )
if rank == 0:
    logger.info("RNA process start")
    
all_genes = adata.var_names.tolist() + adata_protein.var_names.tolist() + adata_methyl.var_names.tolist()
genes = adata.var_names.tolist()
proteins = adata_protein.var_names.tolist()
methyls = adata_methyl.var_names.tolist()

total_length = 3
modality_patch_counts = {
    'RNA': 1,
    'Protein': 1,
    'METHYL': 1,
}

if rank == 0:
    print(f'RNA dict length: {len(vocab_RNA)}')
    print(f'Protein dict length: {len(vocab_Protein)}')
    print(f'METHYL dict length: {len(vocab_methyl)}')
    
with open(model_config_file_RNA, "r") as f:
    model_configs_RNA = json.load(f)
with open(model_config_file_Protein, "r") as f:
    model_configs_Protein = json.load(f)
with open(model_config_file_methyl, "r") as f:
    model_configs_METHYL = json.load(f)

encoder_dict = {}

encoder_dict['RNA'] = PerformerModel(
    ntoken=len(vocab_RNA),
    d_model=model_configs_RNA["layer_size"],
    nhead=model_configs_RNA["nhead"],
    d_hid=model_configs_RNA["layer_size"],
    nlayers=model_configs_RNA["nlayers"],
    vocab=vocab_RNA,
    pad_token=config.pad_token,
    pad_value=config.pad_value,
    n_input_bins=config.n_bins+2,
    cell_emb_style="cls",
    input_emb_style="category",
    feature_redraw_interval=None,
    auto_check_redraw=False,
    # use_modality_tokens=True
)
encoder_dict['Protein'] = PerformerModel(
    ntoken=len(vocab_Protein),
    d_model=model_configs_Protein["layer_size"],
    nhead=model_configs_Protein["nhead"],
    d_hid=model_configs_Protein["layer_size"],
    nlayers=model_configs_Protein["nlayers"],
    vocab=vocab_Protein,
    pad_token=config.pad_token,
    pad_value=config.pad_value,
    n_input_bins=config.n_bins+2,
    cell_emb_style="cls",
    input_emb_style="category",
    feature_redraw_interval=None,
    auto_check_redraw=False,
    # use_modality_tokens=True
)
encoder_dict['METHYL'] = PerformerModel(
    ntoken=len(vocab_methyl),
    d_model=model_configs_METHYL["layer_size"],
    nhead=model_configs_METHYL["nhead"],
    d_hid=model_configs_METHYL["layer_size"],
    nlayers=model_configs_METHYL["nlayers"],
    vocab=vocab_methyl,
    pad_token=config.pad_token,
    pad_value=config.pad_value,
    n_input_bins=config.n_bins+2,
    cell_emb_style="cls",
    input_emb_style="category",
    feature_redraw_interval=None,
    auto_check_redraw=False,
    # use_modality_tokens=True
)

model_dict = encoder_dict['RNA'].state_dict()
pretrained_dict = torch.load(model_RNA_file, map_location=device)
pretrained_dict = {
    k: v
    for k, v in pretrained_dict.items()
    if k in model_dict and v.shape == model_dict[k].shape
}
model_dict.update(pretrained_dict)
encoder_dict['RNA'].load_state_dict(model_dict)
encoder_dict['RNA'].to(device)

model_dict = encoder_dict['Protein'].state_dict()
pretrained_dict = torch.load(model_Protein_file, map_location=device)
pretrained_dict = {
    k: v
    for k, v in pretrained_dict.items()
    if k in model_dict and v.shape == model_dict[k].shape
}
model_dict.update(pretrained_dict)
encoder_dict['Protein'].load_state_dict(model_dict)
encoder_dict['Protein'].to(device)

# METHYL
model_dict = encoder_dict['METHYL'].state_dict()
pretrained_dict = torch.load(model_methyl_file, map_location=device)
pretrained_dict = {
    k: v
    for k, v in pretrained_dict.items()
    if k in model_dict and v.shape == model_dict[k].shape
}
model_dict.update(pretrained_dict)
encoder_dict['METHYL'].load_state_dict(model_dict)
encoder_dict['METHYL'].to(device)


vocab_mod = config.modality_dict.copy()
vocab_mod.update({'<pad>': 3})

# 或者一行代码
vocab_mod = {**config.modality_dict, '<pad>': 3}
full_modality_index = 0
fusion_model = OmniFusionBlock(config.num_modalities, full_modality_index, total_length, model_configs_Protein["layer_size"], config.num_layers_fus, config.num_experts, config.num_routers, config.top_k, model_configs_Protein["nhead"], config.dropout, vocab_mod=vocab_mod, device=device).to(device)


fusion_model = DDP(fusion_model, device_ids=[local_rank], find_unused_parameters=True, static_graph=True)
for key in encoder_dict:
    encoder_dict[key] = DDP(encoder_dict[key], device_ids=[local_rank], find_unused_parameters=True)

criterion_gep_gepc = masked_mse_loss
criterion_simsiam = nn.CosineSimilarity(dim=1)
n_full_modalities = len(config.modality_dict)

unique_patch_counts = set(modality_patch_counts.values())

# shape: (2^n-1, n, length, hidden_dim)
# 2^n-1代表存在模态的情况。当三个模态，只存在一个模态时候，n_full_modalities的作用就是用来指定补充A或者补充B模态。
missing_embeds_dict = {}
num_combinations = (2**config.num_modalities) - 1 
for key, count in modality_patch_counts.items():
    initial_tensor = torch.randn(
        num_combinations, config.num_modalities, count, config.layer_size,
        device=device
    )
    missing_embeds_dict[key] = torch.nn.Parameter(initial_tensor, requires_grad=True)

if config.frozen_encoder:
    if rank == 0:
        logger.info("Freezing encoder parameters...")
    for modality, encoder in encoder_dict.items():
        # ... 原有的冻结逻辑 ...
        for name, param in encoder.named_parameters():
            if 'decoder' in name or 'mvc_decoder' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    # 1. 获取所有需要训练的 Encoder 参数
    params_to_train = []
    for modality, encoder in encoder_dict.items():
        # 筛选出 requires_grad = True 的参数
        trainable_encoder_params = [p for p in encoder.parameters() if p.requires_grad]
        params_to_train.extend(trainable_encoder_params)

        # (可选) 打印出哪些参数是可训练的，用于调试
        if rank == 0:  # 只在主进程打印
            logger.info('using for finetuning')
            for name, param in encoder.named_parameters():
                if param.requires_grad:
                    logger.info(f"  - {name}")
    params_to_train.extend(list(fusion_model.parameters()))
    params_to_train.extend(list(missing_embeds_dict.values()))
    optimizer = torch.optim.Adam(
        params_to_train, lr=config.lr, eps=1e-4 if config.amp else 1e-8
    )
else:
    params = list(fusion_model.parameters()) + [param for encoder in encoder_dict.values() for param in encoder.parameters()]
    for me in missing_embeds_dict.values():
        params += [me]
    optimizer = torch.optim.Adam(
        params, lr=config.lr, eps=1e-4 if config.amp else 1e-8
    )

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=config.schedule_ratio)
scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

best_val_loss = float("inf")
best_model = None

if rank == 0:
    define_wandb_metrcis()

modalities_data = {
    'RNA': {
        'file_path': RNA_file,      # <--- 【新增】
        'gene_ids': gene_ids_RNA,   # <--- 【新增】
        'vocab': vocab_RNA,
        'adata': adata,
        'char': 'R' # 代表 RNA 的字符
    },
    'Protein': {
        'file_path': Protein_file,     # <--- 【新增】
        'gene_ids': proteins_ids_Protein,  # <--- 【新增】
        'vocab': vocab_Protein,
        'adata': adata_protein,
        'char': 'P' # 代表 Protein 的字符
    },
    'METHYL': {
        'file_path': methyl_file,   # <--- 【新增】
        'gene_ids': gene_ids_METHYL,# <--- 【新增】
        'vocab': vocab_methyl,
        'adata': adata_methyl,
        'char': 'M' # 代表 Methylation 的字符
    }
}

if rank == 0:
    (
        train_data_dict,
        valid_data_dict,
        observed_idx_train,
        observed_idx_valid,
    ) = create_flexmoe_input(all_ptid, modalities_data, config, rank, save_path=save_dir)
else:
    (
        train_data_dict,
        valid_data_dict,
        observed_idx_train,
        observed_idx_valid,
    ) = create_flexmoe_input(all_ptid, modalities_data, config, rank)

for d in [train_data_dict, valid_data_dict]:
    d['RNA_gene_ids'] = gene_ids_RNA
    d['Protein_gene_ids'] = proteins_ids_Protein
    d['METHYL_gene_ids'] = gene_ids_METHYL

omics_gene_ids = {
    'RNA': gene_ids_RNA,
    'Protein': proteins_ids_Protein,
    'METHYL': gene_ids_METHYL
}
my_collator = DataCollator(config, vocab_dict, omics_gene_ids)

# 3. (可选优化) Dataset 其实只需要实例化一次，建议移到循环外
train_dataset = MultiModalDataset(
    data_dict=train_data_dict,
    observed_idx=observed_idx_train,
    config=config,
    vocab_dict=vocab_dict
)

valid_dataset = MultiModalDataset(
    data_dict=valid_data_dict,
    observed_idx=observed_idx_valid,
    config=config,
    vocab_dict=vocab_dict
)

# ==================== 【新增修改】Step 1: 在循环外初始化 Momentum 模型 ====================
# 必须在这里初始化，保证它们在整个训练过程中持续存在，保留历史记忆
momentum_encoder_dict = {}
for key, module in encoder_dict.items():
    # 处理 DDP 包装
    base_model = module.module if isinstance(module, torch.nn.parallel.DistributedDataParallel) else module
    m_encoder = copy.deepcopy(base_model)
    m_encoder.to(device)
    m_encoder.train() # 保持 train 模式以更新 BN 统计量
    for param in m_encoder.parameters():
        param.requires_grad = False
    momentum_encoder_dict[key] = m_encoder

base_fusion = fusion_model.module if isinstance(fusion_model, torch.nn.parallel.DistributedDataParallel) else fusion_model
momentum_fusion_model = copy.deepcopy(base_fusion)
momentum_fusion_model.to(device)
momentum_fusion_model.train()
for param in momentum_fusion_model.parameters():
    param.requires_grad = False


for epoch in range(1, config.epochs + 1):

    epoch_start_time = time.time()

    # (Dataset 已经移到循环外实例化了，这里不需要再重复 new 了)
    
    # 设置 Sampler 的 epoch 以保证 shuffle 随机性
    if epoch > config.ranking_epoch:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_sampler.set_epoch(epoch) 

    # === 【关键修改】使用 my_collator ===
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        collate_fn=my_collator,  # <--- 这里改用实例
        sampler=train_sampler, 
        pin_memory=True,
        num_workers=4
    )

    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    val_loader = DataLoader(
        valid_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        collate_fn=my_collator,  # <--- 这里改用实例
        sampler=valid_sampler, 
        pin_memory=True,
        num_workers=4
    )

    train_sampler.set_epoch(epoch)  # 确保每个epoch的shuffle都不同

    if config.do_train:
        train(
            loader=train_loader,
            encoder_dict=encoder_dict,
            fusion_model=fusion_model,
            momentum_encoder_dict=momentum_encoder_dict,
            momentum_fusion_model=momentum_fusion_model,
            missing_embeds_dict=missing_embeds_dict,
            modality_patch_counts=modality_patch_counts,
            vocab_dict=vocab_dict,
            criterion_gep_gepc=criterion_gep_gepc,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config,
            logger=logger,
            epoch=epoch,
            rank=rank,
            vocab_mod=vocab_mod
        )

    val_loss = evaluate(
        loader=val_loader,
        encoder_dict=encoder_dict,
        fusion_model=fusion_model,
        missing_embeds_dict=missing_embeds_dict,
        modality_patch_counts=modality_patch_counts,
        vocab_dict=vocab_dict,
        criterion_gep_gepc=criterion_gep_gepc,
        device=device,
        config=config,
        epoch=epoch,
        logger=logger,
        rank=rank,
        vocab_mod=vocab_mod
    )

    elapsed = time.time() - epoch_start_time
    if rank == 0:
        logger.info("-" * 89)
        logger.info(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss {val_loss:5.4f} | "
        )
        logger.info("-" * 89)

    # if val_loss < best_val_loss:
    # best_val_loss = val_loss
    best_model_epoch = epoch

    if rank == 0:
        # logger.info(f"New best model found at epoch {best_model_epoch} with validation loss {val_loss:.4f}")
        logger.info(f"Validation loss {val_loss:.4f}")
        logger.info(f"Saving model to {save_dir}")

        # 【修改1】: 通过 .module 访问原始模型来获取 state_dict
        best_model_fus_sd = copy.deepcopy(fusion_model.module.state_dict())
        best_model_enc_sd = {
            modality: copy.deepcopy(encoder.module.state_dict())
            for modality, encoder in encoder_dict.items()
        }

        best_model_me_sd = {
            key: param.cpu().data.clone()
            for key, param in missing_embeds_dict.items()
        }

        save_path = save_dir / f"model_e{best_model_epoch}.pt"

        # 1. 保存模型
        torch.save({
            'missing_embeds': best_model_me_sd,
            'fusion_model': best_model_fus_sd,
            'encoder_dict': best_model_enc_sd,
        }, save_path)

        vocab_file_check = save_dir / "vocab_rna.json"
        if not vocab_file_check.exists():
            # 2. 保存词汇表 (vocab.json)
            with open(save_dir / "vocab_protein.json", "w") as f:
                json.dump(vocab_Protein.get_stoi(), f, indent=4)
            with open(save_dir / "vocab_rna.json", "w") as f:
                json.dump(vocab_RNA.get_stoi(), f, indent=4)
            with open(save_dir / "vocab_methyl.json", "w") as f:
                json.dump(vocab_methyl.get_stoi(), f, indent=4)

            # 3. 保存超参数 (args.json)
            args_file = save_dir / "args.json"
            # 如果 config 是 wandb.config 对象，需要先转为字典
            args_dict = dict(config)
            with open(args_file, "w") as f:
                json.dump(args_dict, f, indent=2)

    scheduler.step()
if rank == 0:
    run.finish()
    wandb.finish()
dist.destroy_process_group()
gc.collect()
