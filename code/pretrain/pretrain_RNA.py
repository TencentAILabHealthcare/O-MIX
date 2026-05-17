import copy
import gc
import json
import os
import sys
import time
import traceback
import warnings
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import torch
import torch.distributed as dist
import torchtext
import wandb
from anndata import AnnData
from omix.tokenizer.gene_tokenizer import GeneVocab
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchtext._torchtext import Vocab as VocabPybind
# torchtext.disable_torchtext_deprecation_warning()
from torchtext.vocab import Vocab
from wandb.sdk.wandb_config import Config

sys.path.append("../")
import h5py
import pandas as pd
import omix as omix_logger
from omix import DistributedSubsetsBatchSampler, SubsetsBatchSampler
from omix.loss import (criterion_neg_log_bernoulli, masked_mse_loss,
                        masked_relative_error)
from omix.model import PerformerModel
from omix.preprocess_bulk import Preprocessor
from omix.tokenizer import random_mask_value, tokenize_and_pad_batch
from omix.utils import category_str2int, eval_scib_metrics, set_seed

sc.set_figure_params(figsize=(4, 4))
os.environ["KMP_WARNINGS"] = "off"
# os.environ["WANDB_MODE"] = "offline"

hyperparameter_defaults = dict(
    seed=42,
    # dataset_name="PBMC_10K",
    dataset_name="rna_pretrain",
    do_train=True,
    load_model=None, # "examples/save/omix_human", None
    mask_ratio=0.15,
    epochs=20,
    n_bins=51,
    GEPC=True,  # Masked value prediction for cell embedding
    ecs_thres=0.0,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    dab_weight=0.0,
    lr=1e-4,
    batch_size=8,
    # batch_size=2,
    layer_size=512,
    nlayers=6,
    nhead=8,
    # if load model, batch_size, layer_size, nlayers, nhead will be ignored
    dropout=0.2,
    schedule_ratio=0.9,  # ratio of epochs for learning rate schedule
    save_eval_interval=1,
    log_interval=20,
    fast_transformer=False,
    pre_norm=False,
    amp=False,  # Automatic Mixed Precision
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
    config = Config()
    config.update(hyperparameter_defaults)
    # config = hyperparameter_defaults

set_seed(config.seed)

# %%
# settings for input and preprocessing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
mask_ratio = config.mask_ratio
n_input_bins = config.n_bins
mask_value = n_input_bins + 1
pad_value = n_input_bins


n_hvg = 19205  # number of highly variable genes
# n_hvg = 19282
max_seq_len = n_hvg + 1
per_seq_batch_sample = False
DSBN = False  # Domain-spec batchnorm
explicit_zero_prob = False  # whether explicit bernoulli for zeros

dataset_name = config.dataset_name
save_dir = Path(f"./save/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"save to {save_dir}")
# save the whole script to the dir
os.system(f"cp {__file__} {save_dir}")

logger = omix_logger.logger
omix_logger.utils.add_file_handler(logger, save_dir / "run.log")
logger.info(f'DSBN {DSBN}')
print('start loading...')

import anndata

adata_path = '../../data/pretraining_data/rna_full_data.h5ad'
# load matrix in backend
tpm_f = h5py.File(adata_path, "r")
expression_matrix = tpm_f['X']

adata_tmp = anndata.read_h5ad(adata_path, backed='r')

# load into memory
sample_ids = adata_tmp.obs_names.copy().to_numpy()
# gene_names = adata_tmp.var["gene_name"].copy().tolist()
gene_names = adata_tmp.var_names.copy().tolist()

adata = AnnData(X=expression_matrix, obs=pd.DataFrame(index=sample_ids))
adata.var_names = gene_names  # 将基因名设置为 var_names
adata.var["gene_name"] = gene_names

gene_names = adata.var["gene_name"].tolist()
adata.var_names = gene_names  # 将基因名设置为 var_names
adata.var["gene_name"] = adata.var.index.tolist()
print("过滤前样本数量:", adata.n_obs)


print('okk')
if config.load_model is not None:
    model_dir = Path(config.load_model)
    vocab_file = model_dir / "pc_gene_vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    adata = adata[:, adata.var["id_in_vocab"] >= 0]

    if rank == 0:
        logger.info(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}."
        )

embsize = config.layer_size
nhead = config.nhead
nlayers = config.nlayers
d_hid = config.layer_size

if rank == 0:
    print('copying data')

genes = adata.var["gene_name"].tolist()


if rank == 0:
    print('start splitting')
# 1. 创建一个代表所有细胞索引的数组
n_obs = adata.shape[0]
indices = np.arange(n_obs)

# 2. 对索引进行分割和重排。这个操作非常快，因为只处理一个一维整数数组。
train_idx, valid_idx = train_test_split(
    indices,
    test_size=0.1,
    shuffle=True,
    random_state=42
)
if rank == 0:
    print('extracting from disk')
# 3. 使用这些高效生成的索引来从原始数据中提取训练集和验证集
#    这个切片操作比重排整个矩阵要快得多。

train_idx = np.sort(train_idx)
valid_idx = np.sort(valid_idx)
if rank == 0:
    print('sort index finished')

if rank == 0:
    print('splitting finish')

if config.load_model is None:
    vocab = Vocab(
        VocabPybind(genes + special_tokens, None)
    )  # bidirectional lookup [gene <-> int]
vocab.set_default_index(vocab["<pad>"])
gene_ids = np.array(vocab(genes), dtype=int)


class LazyExpressionDataset(Dataset):
    """
    按需读取磁盘上的 H5AD 数据
    """
    def __init__(self, data_path: str, indices: np.ndarray, gene_names: List[str]):
        self.data_path = data_path
        self.indices = indices
        self.gene_names = gene_names
        self.data_file = None
        self.data_matrix = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 这种写法是为了兼容多进程 num_workers > 0
        # 每个 worker 进程里第一次调用时打开文件
        if self.data_matrix is None:
            # 这里假设 data_path 是 h5ad，底层是 hdf5
            # 通常 h5ad 的表达量存在 /X 中，如果是 csr_matrix 可能会在 /X/data 等中
            # 为了简单通用，我们用 h5py 直接读 'X' 节点，假设是 dense 或者 h5py 能处理的格式
            self.data_file = h5py.File(self.data_path, 'r')
            self.data_matrix = self.data_file['X']
        
        real_idx = self.indices[idx]
        # 读取一行数据
        expression = self.data_matrix[real_idx]
        
        # 如果是稀疏矩阵存储在 h5py 中，可能需要特殊处理 decode
        # 假设这里是 dense float array
        if hasattr(expression, "toarray"):
            expression = expression.toarray().flatten()
        elif isinstance(expression, np.ndarray):
            expression = expression.flatten()
            
        return expression

def custom_collate_fn(batch, gene_ids, vocab, max_len, pad_token, pad_value, mask_ratio, mask_value):
    """
    在每个 Batch 读取后，进行 tokenize 和 masking
    """
    # 1. Stack batch data (NumPy arrays)
    # batch 是 list of numpy arrays
    batch_data = np.vstack(batch) 
    
    # 2. Tokenize and Pad
    # 这个函数现在只处理一个 batch 的数据，内存消耗很小
    tokenized_data = tokenize_and_pad_batch(
        batch_data,
        gene_ids,
        max_len=max_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,
        include_zero_gene=False,
    )
    
    # 3. Random Masking
    masked_values = random_mask_value(
        tokenized_data["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    
    # 4. Construct Output Dict
    return {
        "gene_ids": tokenized_data["genes"],
        "values": masked_values,         # Masked values as input
        "target_values": tokenized_data["values"], # Original values as target
    }

# 实例化 Dataset
train_ds = LazyExpressionDataset(adata_path, train_idx, genes)
valid_ds = LazyExpressionDataset(adata_path, valid_idx, genes)

if rank == 0:
    logger.info(f"Train samples: {len(train_ds)}, Valid samples: {len(valid_ds)}")

# 准备 Collate Function (使用 partial 固定参数)
collate_func = partial(
    custom_collate_fn,
    gene_ids=gene_ids,
    vocab=vocab,
    max_len=max_seq_len,
    pad_token=pad_token,
    pad_value=pad_value,
    mask_ratio=mask_ratio,
    mask_value=mask_value
)

# 准备 DataLoaders
def prepare_dataloader(dataset, batch_size, shuffle, drop_last, sampler_seed):
    sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last, seed=sampler_seed)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4, # 可以开启多进程读取，因为我们在 Dataset 内部处理了文件打开
        collate_fn=collate_func, # 关键：在这里进行处理
        pin_memory=True,
        drop_last=drop_last
    )
    return loader

ntokens = len(vocab)  # size of vocabulary
model = PerformerModel(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    vocab=vocab,
    dropout=config.dropout,
    pad_token=pad_token,
    pad_value=pad_value,
    do_mvc=config.GEPC,
    do_dab=False,
    use_batch_labels=DSBN,
    num_batch_labels=None,
    domain_spec_batchnorm=DSBN,
    n_input_bins=n_input_bins+2,
    ecs_threshold=config.ecs_thres,
    explicit_zero_prob=explicit_zero_prob,
    use_fast_transformer=config.fast_transformer,
    pre_norm=config.pre_norm,
    input_emb_style="category"
)


model.to(device)
model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

if rank == 0:
    wandb.watch(model)


criterion = masked_mse_loss
# criterion_dab = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=config.lr, eps=1e-4 if config.amp else 1e-8
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=config.schedule_ratio)

scaler = torch.cuda.amp.GradScaler(enabled=config.amp)


def train(model: nn.Module, loader: DataLoader, epoch: int) -> None:
    model.train()
    total_loss, total_mse, total_gepc = 0.0, 0.0, 0.0
    total_error = 0.0
    log_interval = config.log_interval
    start_time = time.time()

    num_batches = len(loader)
    for batch, batch_data in enumerate(loader):
        # 注意：collate_fn 返回的已经是处理好的 Tensor 了
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)

        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        
        with torch.cuda.amp.autocast(enabled=config.amp):
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=None,
                MVC=config.GEPC,
                ECS=config.ecs_thres > 0,
            )
            
            masked_positions = input_values.eq(mask_value)
            loss = loss_mse = criterion(
                output_dict["mlm_output"], target_values, masked_positions
            )
            
            if config.GEPC:
                loss_gepc = criterion(
                    output_dict["mvc_output"], target_values, masked_positions
                )
                loss = loss + loss_gepc

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Metrics logging (Same as before)
        with torch.no_grad():
            mre = masked_relative_error(
                output_dict["mlm_output"], target_values, masked_positions
            )

        total_loss += loss.item()
        total_mse += loss_mse.item()
        total_gepc += loss_gepc.item() if config.GEPC else 0.0
        total_error += mre.item()
        
        if batch % log_interval == 0 and batch > 0 and rank == 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            cur_gepc = total_gepc / log_interval if config.GEPC else 0.0
            cur_error = total_error / log_interval
            
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:08.7f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} | mre {cur_error:5.2f} |"
                + (f"gepc {cur_gepc:5.2f} |" if config.GEPC else "")
            )
            
            wandb.log({
                "train/mse": cur_mse,
                "train/mre": cur_error,
                "train/loss": cur_loss
            })

            total_loss = 0
            total_mse = 0
            total_gepc = 0
            total_error = 0
            start_time = time.time()

def evaluate(model: nn.Module, loader: DataLoader) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    total_num = 0
    
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None,
                )
                output_values = output_dict["mlm_output"]
                masked_positions = input_values.eq(mask_value)
                loss = criterion(output_values, target_values, masked_positions)

            total_loss += loss.item() * len(input_gene_ids)
            total_error += masked_relative_error(output_values, target_values, masked_positions).item() * len(input_gene_ids)
            total_num += len(input_gene_ids)

    # DDP Reduce logic
    total_loss_tensor = torch.tensor([total_loss], dtype=torch.float64, device=device)
    total_error_tensor = torch.tensor([total_error], dtype=torch.float64, device=device)
    total_num_tensor = torch.tensor([total_num], dtype=torch.float64, device=device)

    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_error_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_num_tensor, op=dist.ReduceOp.SUM)

    final_loss = total_loss_tensor.item() / total_num_tensor.item()
    final_mre = total_error_tensor.item() / total_num_tensor.item()

    if rank == 0:
        wandb.log({
            "valid/mse": final_loss,
            "valid/mre": final_mre,
            "epoch": epoch,
        })

    return final_loss, final_mre


def define_wandb_metrcis():
    wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mre", summary="min", step_metric="epoch")
    # wandb.define_metric("valid/dab", summary="min", step_metric="epoch")
    # wandb.define_metric("valid/sum_mse_dab", summary="min", step_metric="epoch")
    wandb.define_metric("test/avg_bio", summary="max")


# %%
best_val_loss = float("inf")
best_avg_bio = 0.0
best_model = None
if rank == 0:
    define_wandb_metrcis()

for epoch in range(1, config.epochs + 1):
    epoch_start_time = time.time()
    train_loader = prepare_dataloader(train_ds, config.batch_size, shuffle=True, drop_last=False, sampler_seed=epoch)
    valid_loader = prepare_dataloader(valid_ds, config.batch_size, shuffle=False, drop_last=False, sampler_seed=epoch)

    if config.do_train:
        train(model, train_loader, epoch)
    val_loss, val_mre = evaluate(model, valid_loader)
    elapsed = time.time() - epoch_start_time
    if rank == 0:
        logger.info("-" * 89)
        logger.info(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss/mse {val_loss:5.4f} | mre {val_mre:5.4f}"
        )
        logger.info("-" * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        if rank == 0:
            best_model = copy.deepcopy(model.module)  # 保存 unwrapped model
            best_model_epoch = epoch
            logger.info(f"Best model with score {best_val_loss:5.4f}")

            logger.info(f"Saving model to {save_dir}")
            torch.save(best_model.state_dict(), save_dir / f"model_e{best_model_epoch}.pt")

            vocab_file_check = save_dir / "vocab.json"
            if not vocab_file_check.exists():
                # 2. 保存词汇表 (vocab.json)
                vocab_file = save_dir / "vocab.json"
                with open(vocab_file, "w") as f:
                    # json.dump(vocab.get_itos(), f) # list
                    json.dump(vocab.get_stoi(), f, indent=4)  # dict

                # 3. 保存超参数配置 (args.json)
                args_file = save_dir / "args.json"
                args_dict = dict(config)
                with open(args_file, "w") as f:
                    json.dump(args_dict, f, indent=2)

    scheduler.step()


if rank == 0:

    # run.log_artifact(artifact)
    run.finish()
    wandb.finish()

dist.destroy_process_group()
gc.collect()
