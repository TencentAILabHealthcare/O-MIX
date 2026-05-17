# %%
"""
Hyper (multi-checkpoint) RNA clustering & visualization.

For a single pretrained run, sweep over a list of training-epoch checkpoints
(e.g. epochs 1 / 5 / 10 / 15 / 20) and, for each checkpoint, run the downstream
clustering / visualization pipeline with multiple random seeds. ARI / NMI /
Silhouette are reported as mean ± std per epoch.

Data loading, tokenization and model construction are performed ONCE; only
``state_dict`` is swapped between epochs, and embeddings are recomputed from
that. Figures are saved under ``code/clustering/save_figs``.
"""
import json
import os
import random
import sys
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)

sys.path.append("../")
from omix.model.model_uni import PerformerModel
from omix.preprocess_bulk import Preprocessor
from omix.tokenizer import GeneVocab, tokenize_and_pad_batch

LOAD_MODEL_PATH = "../pretrain/save/rna_pretrain"

DISEASE_NAME_LIST = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC', 'KICH',
                     'KIRC', 'KIRP',  'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'READ', 'SARC', 'SKCM',
                     'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS']

DATA_ROOT_DIR = "../../data/GDAC"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
N_BINS = 51
PAD_TOKEN = "<pad>"
PAD_VALUE = N_BINS
MASK_VALUE = N_BINS + 1
MAX_SEQ_LEN = 19205 + 1
load_model = True
cell_emb_style = "cls"
dropout = 0.0

# Multi-run configuration
SEEDS = [100, 101, 102, 103, 104]
EPOCHS_TO_TEST = [1, 5, 10, 15, 20]
FIG_DIR = Path("save_figs")


def set_global_seed(seed: int):
    """Seed all relevant RNGs for reproducibility of a single run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_combine_data(disease_list: list, data_dir: str) -> anndata.AnnData:
    """Load multiple cancer datasets, tag each with its pancancer type, and concatenate."""
    used_gene_file = '../../data/GDAC/filtered_genes_list.csv'
    gene_list_df = pd.read_csv(used_gene_file)
    genes_to_keep = gene_list_df['gene_name'].tolist()

    adata_list = []
    print("Loading and combining data...")
    for disease in disease_list:
        print(f"-> Loading {disease}...")
        disease_path = os.path.join(data_dir, disease)
        if not os.path.exists(disease_path):
            print(f"Warning: Directory for {disease} not found. Skipping.")
            continue

        labels = pd.read_csv(os.path.join(disease_path, 'labels.csv'), sep='\t', index_col=0)
        mrna = pd.read_csv(os.path.join(disease_path, 'mRNA_TPM.csv'), index_col=0)
        methyl = pd.read_csv(os.path.join(disease_path, 'methylation.csv'), index_col=0)
        protein = pd.read_csv(os.path.join(disease_path, 'RPPA.csv'), index_col=0)

        genes_present_in_file = mrna.columns
        common_genes = [gene for gene in genes_to_keep if gene in genes_present_in_file]
        mrna = mrna[common_genes]

        mrna.index = mrna.index.str.slice(0, 12).str.lower()
        mrna = mrna[~mrna.index.duplicated(keep='first')]

        protein.index = protein.index.str.slice(0, 12).str.lower()
        protein = protein[~protein.index.duplicated(keep='first')]

        methyl.index = methyl.index.str.slice(0, 12).str.lower()
        methyl = methyl[~methyl.index.duplicated(keep='first')]

        common_samples = mrna.index.intersection(protein.index).intersection(methyl.index)
        if disease == 'COAD' or disease == 'READ':
            disease = 'COADREAD'

        if len(common_samples) == 0:
            print(f"Warning: No common samples found for {disease}. Skipping.")
            continue
        mrna = mrna.loc[common_samples]

        adata = anndata.AnnData(X=mrna)
        adata.var_names_make_unique()
        adata.var["gene_name"] = adata.var.index.tolist()

        adata.obs['pancancer_type'] = disease

        adata_list.append(adata)

    if not adata_list:
        raise ValueError("No data could be loaded. Please check your paths and disease list.")

    combined_adata = anndata.concat(adata_list, join='outer', label='batch', index_unique=None)
    combined_adata.obs_names_make_unique()

    print("Data loading and combination complete.")
    print("Combined data shape:", combined_adata.shape)
    print("Pancancer types loaded:", combined_adata.obs['pancancer_type'].unique().tolist())
    combined_adata.X = np.nan_to_num(combined_adata.X, nan=0.0)
    return combined_adata


def preprocess_and_tokenize(adata: anndata.AnnData, vocab: GeneVocab):
    """Run preprocessing + tokenization once; reusable across checkpoints."""
    preprocessor = Preprocessor(
        use_key="X",
        filter_gene_by_counts=False,
        filter_cell_by_counts=False,
        normalize_total=False,
        log1p=True,
        binning=N_BINS,
        result_binned_key="X_binned",
        repair=False,
    )
    preprocessor(adata)

    input_data = adata.layers["X_binned"]
    gene_ids = np.array(vocab(adata.var.index.tolist()), dtype=int)
    tokenized_data = tokenize_and_pad_batch(
        input_data,
        gene_ids,
        max_len=MAX_SEQ_LEN,
        vocab=vocab,
        pad_token=PAD_TOKEN,
        pad_value=PAD_VALUE,
        append_cls=True,
        include_zero_gene=False,
    )

    all_gene_ids = tokenized_data['genes']
    all_values = tokenized_data['values']
    src_key_padding_mask = all_gene_ids.eq(vocab[PAD_TOKEN])
    return all_gene_ids, all_values, src_key_padding_mask


def encode_with_model(model: PerformerModel, all_gene_ids, all_values, src_key_padding_mask) -> np.ndarray:
    """Run model.encode_batch and L2-normalize. Assumes model is already in eval mode on DEVICE."""
    print("Generating embeddings for all samples...")
    device = next(model.parameters()).device
    print(device)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        cell_embeddings = model.encode_batch(
            all_gene_ids,
            all_values.float(),
            src_key_padding_mask=src_key_padding_mask,
            batch_size=BATCH_SIZE,
            time_step=0,
            return_np=True,
        )

    cell_embeddings = np.array(cell_embeddings)
    norm = np.linalg.norm(cell_embeddings, axis=1, keepdims=True)
    cell_embeddings = cell_embeddings / (norm + 1e-6)
    return cell_embeddings


def load_checkpoint_into_model(model: PerformerModel, ckpt_path: Path):
    """Replace model weights with those from ``ckpt_path`` (shape-matching keys only)."""
    print(f"Loading checkpoint from {ckpt_path}")
    pretrained_dict = torch.load(ckpt_path, map_location=DEVICE)
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    print(f"  Matched {len(pretrained_dict)} / {len(model_dict)} parameter tensors.")
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()


def run_single_clustering(adata: anndata.AnnData, embeddings: np.ndarray, seed: int,
                          epoch: int, fig_dir: Path) -> dict:
    """Run neighbors + leiden + t-SNE with a given seed, return metrics and save figure."""
    set_global_seed(seed)

    # Work on a shallow copy so each (epoch, seed) is independent.
    adata_run = adata.copy()
    adata_run.obsm['X_omix'] = embeddings

    print(f"\n=== [epoch={epoch}, seed={seed}] Clustering & Metrics ===")
    sc.pp.neighbors(adata_run, use_rep='X_omix', n_neighbors=30, random_state=seed)
    sc.tl.leiden(adata_run, resolution=1.0, key_added='leiden_clusters', random_state=seed)

    true_labels = adata_run.obs['pancancer_type']
    predicted_labels = adata_run.obs['leiden_clusters']

    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    silhouette = silhouette_score(embeddings, predicted_labels)

    print(f"[epoch={epoch}, seed={seed}] ARI: {ari:.4f}")
    print(f"[epoch={epoch}, seed={seed}] NMI: {nmi:.4f}")
    print(f"[epoch={epoch}, seed={seed}] Silhouette: {silhouette:.4f}")

    print(f"[epoch={epoch}, seed={seed}] Visualizing with t-SNE...")
    sc.tl.tsne(adata_run, use_rep='X_omix', random_state=seed)

    fig = sc.pl.tsne(
        adata_run,
        color='pancancer_type',
        title=f't-SNE of RNA Embeddings by Cancer Type (epoch={epoch}, seed={seed})',
        show=False,
        return_fig=True,
    )
    fig_path = fig_dir / f'embeddings_rna_pretrain{load_model}_e{epoch}_seed{seed}.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"[epoch={epoch}, seed={seed}] t-SNE plot saved to {fig_path}")

    return {
        'epoch': int(epoch),
        'seed': int(seed),
        'ari': float(ari),
        'nmi': float(nmi),
        'silhouette': float(silhouette),
    }


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading pre-trained model and vocab...")
    model_dir = Path(LOAD_MODEL_PATH)
    model_config_file = model_dir / "args.json"
    vocab_file = model_dir / "vocab.json"

    # Validate every requested checkpoint up-front so we fail fast.
    ckpt_files = {epoch: model_dir / f"model_e{epoch}.pt" for epoch in EPOCHS_TO_TEST}
    missing = [str(p) for p in ckpt_files.values() if not p.exists()]
    if missing or not model_config_file.exists() or not vocab_file.exists():
        raise FileNotFoundError(
            f"Missing files. config_ok={model_config_file.exists()}, "
            f"vocab_ok={vocab_file.exists()}, missing_ckpts={missing}"
        )

    special_tokens = [PAD_TOKEN, "<cls>", "<eoc>"]
    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    with open(model_config_file, "r") as f:
        model_configs = json.load(f)

    # Seed once for model construction / data loading; per-run seeds are set inside the loop.
    set_global_seed(SEEDS[0])

    model = PerformerModel(
        ntoken=len(vocab),
        d_model=model_configs["layer_size"],
        nhead=model_configs["nhead"],
        d_hid=model_configs["layer_size"],
        nlayers=model_configs["nlayers"],
        vocab=vocab,
        pad_token=PAD_TOKEN,
        pad_value=PAD_VALUE,
        n_input_bins=N_BINS + 2,
        cell_emb_style=cell_emb_style,
        input_emb_style="category",
    )
    print(f'dropout: {dropout}')
    print(f'cell_emb_style: {cell_emb_style}')
    model.to(DEVICE)
    model.eval()

    # Load + tokenize the data once; reused for every checkpoint.
    adata = load_and_combine_data(DISEASE_NAME_LIST, DATA_ROOT_DIR)
    all_gene_ids, all_values, src_key_padding_mask = preprocess_and_tokenize(adata, vocab)

    all_results = []
    for epoch in EPOCHS_TO_TEST:
        print(f"\n################ Epoch {epoch} ################")
        load_checkpoint_into_model(model, ckpt_files[epoch])

        embeddings = encode_with_model(model, all_gene_ids, all_values, src_key_padding_mask)
        print(f"[epoch={epoch}] embeddings shape:", embeddings.shape)

        for seed in SEEDS:
            metrics = run_single_clustering(adata, embeddings, seed, epoch, FIG_DIR)
            all_results.append(metrics)

    results_df = pd.DataFrame(all_results)
    print("\n================ Per-(epoch, seed) results ================")
    print(results_df.to_string(index=False))

    # Per-epoch aggregation.
    summary_rows = []
    for epoch in EPOCHS_TO_TEST:
        sub = results_df[results_df['epoch'] == epoch]
        ari_mean, ari_std = sub['ari'].mean(), sub['ari'].std(ddof=1)
        nmi_mean, nmi_std = sub['nmi'].mean(), sub['nmi'].std(ddof=1)
        sil_mean, sil_std = sub['silhouette'].mean(), sub['silhouette'].std(ddof=1)
        summary_rows.append({
            'epoch': int(epoch),
            'ari_mean': ari_mean, 'ari_std': ari_std,
            'nmi_mean': nmi_mean, 'nmi_std': nmi_std,
            'silhouette_mean': sil_mean, 'silhouette_std': sil_std,
        })
    summary_df = pd.DataFrame(summary_rows)

    print("\n================ Per-epoch (mean ± std over seeds) ================")
    for _, row in summary_df.iterrows():
        print(
            f"epoch={int(row['epoch']):>2}  "
            f"ARI: {row['ari_mean']:.4f} ± {row['ari_std']:.4f}  "
            f"NMI: {row['nmi_mean']:.4f} ± {row['nmi_std']:.4f}  "
            f"Silhouette: {row['silhouette_mean']:.4f} ± {row['silhouette_std']:.4f}"
        )

    # Persist results next to the figures for downstream inspection.
    csv_path = FIG_DIR / 'multirun_rna_hyper_metrics.csv'
    results_df.to_csv(csv_path, index=False)

    summary_csv_path = FIG_DIR / 'multirun_rna_hyper_metrics_summary.csv'
    summary_df.to_csv(summary_csv_path, index=False)

    summary_txt_path = FIG_DIR / 'multirun_rna_hyper_metrics_summary.txt'
    with open(summary_txt_path, 'w') as f:
        f.write(f"LOAD_MODEL_PATH: {LOAD_MODEL_PATH}\n")
        f.write(f"EPOCHS_TO_TEST: {EPOCHS_TO_TEST}\n")
        f.write(f"SEEDS: {SEEDS}\n\n")
        f.write("Per-(epoch, seed) results:\n")
        f.write(results_df.to_string(index=False) + "\n\n")
        f.write("Per-epoch (mean ± std over seeds):\n")
        for _, row in summary_df.iterrows():
            f.write(
                f"epoch={int(row['epoch']):>2}  "
                f"ARI: {row['ari_mean']:.4f} ± {row['ari_std']:.4f}  "
                f"NMI: {row['nmi_mean']:.4f} ± {row['nmi_std']:.4f}  "
                f"Silhouette: {row['silhouette_mean']:.4f} ± {row['silhouette_std']:.4f}\n"
            )

    print(f"\nPer-(epoch, seed) metrics saved to {csv_path}")
    print(f"Per-epoch summary CSV saved to {summary_csv_path}")
    print(f"Summary text saved to {summary_txt_path}")


if __name__ == '__main__':
    main()
