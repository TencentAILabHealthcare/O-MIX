import copy
import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata
import numpy as np
import scanpy as sc
import torch
import torch.distributed as dist
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
from torchtext.vocab import Vocab

sys.path.append("../")
import h5py
import pandas as pd
import omix as omix_logger
from omix import SubsetsBatchSampler
from omix.loss import (criterion_neg_log_bernoulli, masked_mse_loss,
                        masked_relative_error)
from omix.model import PerformerModel
from omix.preprocess_bulk import Preprocessor
from omix.tokenizer import random_mask_value, tokenize_and_pad_batch
from omix.utils import category_str2int, eval_scib_metrics, set_seed
from wandb.sdk.wandb_config import Config

sc.set_figure_params(figsize=(4, 4))
os.environ["KMP_WARNINGS"] = "off"

hyperparameter_defaults = dict(
    seed=42,
    dataset_name="methylation_pretrain",
    do_train=True,
    load_model=None, # None, examples/save/omix_human
    mask_ratio=0.15,
    # epochs=5,
    epochs=40,
    n_bins=51,
    GEPC=True,  # Masked value prediction for cell embedding
    ecs_thres=0.0,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    dab_weight=0.0,
    # lr=5e-5,
    lr=1e-4,
    batch_size=8,
    layer_size=512,
    nlayers=6,
    nhead=8,
    # if load model, batch_size, layer_size, nlayers, nhead will be ignored
    dropout=0.2,
    schedule_ratio=0.9,  # ratio of epochs for learning rate schedule
    save_eval_interval=1,
    log_interval=50,
    fast_transformer=False,
    pre_norm=False,
    amp=False,  # Automatic Mixed Precision
)
import datetime

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
n_input_bins = config.n_bins
mask_ratio = config.mask_ratio
mask_value = n_input_bins+1
pad_value = n_input_bins

n_hvg = 12920  # number of highly variable genes
max_seq_len = n_hvg + 1
per_seq_batch_sample = False
DSBN = False  # Domain-spec batchnorm
explicit_zero_prob = False  # whether explicit bernoulli for zeros

# %%
dataset_name = config.dataset_name
save_dir = Path(f"./save/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
if rank == 0:
    print(f"save to {save_dir}")
# save the whole script to the dir
os.system(f"cp {__file__} {save_dir}")

logger = omix_logger.logger
omix_logger.utils.add_file_handler(logger, save_dir / "run.log")
if rank == 0:
    logger.info(f'masking ratio: {mask_ratio}')
    print('start loading...')
processed_data_dir = '../../data/pretraining_data/'
# adata_path = os.path.join(processed_data_dir, 'gene_level_methylation_processed.h5ad')
adata_path = os.path.join(processed_data_dir, 'methylation_full_data.h5ad')
adata = anndata.read_h5ad(adata_path, backed='r')
# adata = adata[:10000, :].to_memory()
gene_names = adata.var.index.tolist()
sample_ids = adata.obs.index.tolist()
adata.var_names = gene_names
adata.var["gene_name"] = gene_names
genes = adata.var["gene_name"].tolist()
# 假设数据是原始 counts
data_is_raw = False

if config.load_model is not None:
    model_dir = Path(config.load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    adata = adata[:, adata.var["id_in_vocab"] >= 0]

    # model
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    logger.info(
        f"Resume model from {model_file}, the model args will be overriden by the "
        f"config {model_config_file}."
    )
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]
else:
    embsize = config.layer_size
    nhead = config.nhead
    nlayers = config.nlayers
    d_hid = config.layer_size

n_obs = adata.shape[0]
indices = np.arange(n_obs)

# 2. 对索引进行分割和重排。这个操作非常快，因为只处理一个一维整数数组。
train_idx, valid_idx = train_test_split(
    indices,
    test_size=0.1,
    shuffle=True,
    random_state=42
)

train_idx = np.sort(train_idx)
valid_idx = np.sort(valid_idx)
if rank == 0:
    print('splitting data')
train_data = adata.X[train_idx, :]
valid_data = adata.X[valid_idx, :]

# %%
if config.load_model is None:
    vocab = Vocab(
        VocabPybind(genes + special_tokens, None)
    )  # bidirectional lookup [gene <-> int]
vocab.set_default_index(vocab["<pad>"])
gene_ids = np.array(vocab(genes), dtype=int)
if rank == 0:
    print('tokenizing data')
tokenized_train = tokenize_and_pad_batch(
    train_data,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,  # append <cls> token at the beginning
    include_zero_gene=False,
)
tokenized_valid = tokenize_and_pad_batch(
    valid_data,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,
    include_zero_gene=False,
)
if rank == 0:
    logger.info(
        f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
        f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
    )
    logger.info(
        f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
        f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
    )
    logger.info(f'mask {config.mask_ratio}')


def prepare_data() -> Tuple[Dict[str, torch.Tensor]]:
    masked_values_train = random_mask_value(
        tokenized_train["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    masked_values_valid = random_mask_value(
        tokenized_valid["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    if rank == 0:
        print(
            f"random masking at epoch {epoch:3d}, ratio of masked values in train: ",
            f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}",
        )

    input_gene_ids_train, input_gene_ids_valid = (
        tokenized_train["genes"],
        tokenized_valid["genes"],
    )
    input_values_train, input_values_valid = masked_values_train, masked_values_valid
    target_values_train, target_values_valid = (
        tokenized_train["values"],
        tokenized_valid["values"],
    )

    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "target_values": target_values_train,
    }
    valid_data_pt = {
        "gene_ids": input_gene_ids_valid,
        "values": input_values_valid,
        "target_values": target_values_valid,
    }

    return train_data_pt, valid_data_pt


# dataset
class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


# data_loader
def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
    intra_domain_shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    dataset = SeqDataset(data_pt)
    sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last, seed=seed)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,  # shuffle 必须为 False，因为 Sampler 已经处理了
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader

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


if config.load_model is not None:
    try:
        model.load_state_dict(torch.load(model_file))
        logger.info(f"Loading all model params from {model_file}")
    except:
        # only load params that are in the model and match the size
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        for k, v in pretrained_dict.items():
            logger.info(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

model.to(device)
model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
if rank == 0:
    wandb.watch(model)


criterion = masked_mse_loss

optimizer = torch.optim.Adam(
    model.parameters(), lr=config.lr, eps=1e-4 if config.amp else 1e-8
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=config.schedule_ratio)

scaler = torch.cuda.amp.GradScaler(enabled=config.amp)


def train(model: nn.Module, loader: DataLoader) -> None:
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss, total_mse, total_gepc = 0.0, 0.0, 0.0
    # total_dab = 0.0
    total_error = 0.0
    log_interval = config.log_interval
    start_time = time.time()

    num_batches = len(loader)
    for batch, batch_data in enumerate(loader):
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

            masked_positions = input_values.eq(mask_value)  # the postions to predict
            # 1. reconstruction loss
            loss = loss_mse = criterion(
                output_dict["mlm_output"], target_values, masked_positions
            )
            metrics_to_log = {"train/mse": loss_mse.item()}
            # 2. 通过伯努利分布显式优化“零值预测概率”
            # if explicit_zero_prob:
            #     loss_zero_log_prob = criterion_neg_log_bernoulli(
            #         output_dict["mlm_zero_probs"], target_values, masked_positions
            #     )
            #     loss = loss + loss_zero_log_prob
            #     metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})
            # 3. MVC, Gene expression prediction for cell modeling
            if config.GEPC:
                loss_gepc = criterion(
                    output_dict["mvc_output"], target_values, masked_positions
                )
                loss = loss + loss_gepc
                metrics_to_log.update({"train/mvc": loss_gepc.item()})
            # 4. 通过伯努利分布显式优化“零值预测概率”
            # if config.GEPC and explicit_zero_prob:
            #     loss_gepc_zero_log_prob = criterion_neg_log_bernoulli(
            #         output_dict["mvc_zero_probs"], target_values, masked_positions
            #     )
            #     loss = loss + loss_gepc_zero_log_prob
            #     metrics_to_log.update(
            #         {"train/mvc_nzlp": loss_gepc_zero_log_prob.item()}
            #     )
            # 5. elastic cell similarity优化细胞间或基因间的余弦相似度
            # if config.ecs_thres > 0:
            #     loss_ecs = 10 * output_dict["loss_ecs"]
            #     loss = loss + loss_ecs
            #     metrics_to_log.update({"train/ecs": loss_ecs.item()})
            # 6. batch correction
            # loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
            # loss = loss + config.dab_weight * loss_dab
            # metrics_to_log.update({"train/dab": loss_dab.item()})

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()
        if rank == 0:
            wandb.log(metrics_to_log)

        with torch.no_grad():
            mre = masked_relative_error(
                output_dict["mlm_output"], target_values, masked_positions
            )

        total_loss += loss.item()
        total_mse += loss_mse.item()
        total_gepc += loss_gepc.item() if config.GEPC else 0.0
        # total_dab += loss_dab.item()
        total_error += mre.item()
        if batch % log_interval == 0 and batch > 0 and rank == 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            cur_gepc = total_gepc / log_interval if config.GEPC else 0.0
            cur_error = total_error / log_interval
            # cur_dab = total_dab / log_interval
            # ppl = math.exp(cur_loss)
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:08.7f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:7.6f} | mse {cur_mse:5.2f} | mre {cur_error:5.2f} |"
                + (f"gepc {cur_gepc:5.2f} |" if config.GEPC else "")
            )
            total_loss = 0
            total_mse = 0
            total_gepc = 0
            total_error = 0
            # total_dab = 0
            start_time = time.time()


def define_wandb_metrcis():
    wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mre", summary="min", step_metric="epoch")
    # wandb.define_metric("valid/dab", summary="min", step_metric="epoch")
    # wandb.define_metric("valid/sum_mse_dab", summary="min", step_metric="epoch")
    wandb.define_metric("test/avg_bio", summary="max")


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    # total_dab = 0.0
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
                # loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)

            total_loss += loss.item() * len(input_gene_ids)
            total_error += masked_relative_error(
                output_values, target_values, masked_positions
            ).item() * len(input_gene_ids)
            # total_dab += loss_dab.item() * len(input_gene_ids)
            total_num += len(input_gene_ids)

        # --- 聚合所有进程的指标 ---
        # 1. 将指标转换为 tensor
        total_loss_tensor = torch.tensor([total_loss], dtype=torch.float64, device=device)
        total_error_tensor = torch.tensor([total_error], dtype=torch.float64, device=device)
        # total_dab_tensor = torch.tensor([total_dab], dtype=torch.float64, device=device)
        total_num_tensor = torch.tensor([total_num], dtype=torch.float64, device=device)

        # 2. 使用 all_reduce 进行求和
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_error_tensor, op=dist.ReduceOp.SUM)
        # dist.all_reduce(total_dab_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_num_tensor, op=dist.ReduceOp.SUM)

        # 3. 计算全局平均值
        final_loss = total_loss_tensor.item() / total_num_tensor.item()
        final_mre = total_error_tensor.item() / total_num_tensor.item()
        # final_dab = total_dab_tensor.item() / total_num_tensor.item()
        # ---------------------------

        # --- 日志只在主进程 ---
        if rank == 0:
            wandb.log(
                {
                    "valid/mse": final_loss,
                    "valid/mre": final_mre,
                    # "valid/dab": final_dab,
                    # "valid/sum_mse_dab": (final_loss + config.dab_weight * final_dab),
                    "epoch": epoch,
                },
            )
        # ---------------------

        return final_loss, final_mre


# %%
best_val_loss = float("inf")
best_avg_bio = 0.0
best_model = None
if rank == 0:
    define_wandb_metrcis()

for epoch in range(1, config.epochs + 1):
    epoch_start_time = time.time()
    train_data_pt, valid_data_pt = prepare_data()
    train_loader = prepare_dataloader(
        train_data_pt,
        batch_size=config.batch_size,
        shuffle=True,
        intra_domain_shuffle=True,
        drop_last=False,
    )
    valid_loader = prepare_dataloader(
        valid_data_pt,
        batch_size=config.batch_size,
        shuffle=False,
        intra_domain_shuffle=False,
        drop_last=False,
    )

    if config.do_train:
        train(
            model,
            loader=train_loader,
        )
    val_loss, val_mre = evaluate(
        model,
        loader=valid_loader,
    )
    elapsed = time.time() - epoch_start_time
    if rank == 0:
        logger.info("-" * 89)
        logger.info(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            # f"valid loss/mse {val_loss:5.4f} | mre {val_mre:5.4f} | dab {val_dab:5.4f}"
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
    run.finish()
    wandb.finish()
dist.destroy_process_group()
gc.collect()
