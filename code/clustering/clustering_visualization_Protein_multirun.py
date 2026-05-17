# %%
"""
Multi-run Protein (RPPA) clustering & visualization.

Runs the downstream clustering/visualization pipeline 5 times with different
random seeds on the SAME set of pretrained protein embeddings (model inference
is deterministic in eval mode), and reports ARI / NMI as mean ± std.

Figures are saved under ``code/clustering/save_figs``.
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
from omix.preprocess_rppa import Preprocessor
from omix.tokenizer import GeneVocab, tokenize_and_pad_batch

LOAD_MODEL_PATH = "../pretrain/save/protein_pretrain"

DISEASE_NAME_LIST = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC', 'KICH',
                     'KIRC', 'KIRP',  'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'READ', 'SARC', 'SKCM',
                     'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS']

DATA_ROOT_DIR = "../../data/GDAC"
RENAME_MAP_PATH = '../../data/gene_name_mapping/protein_name_mapping.json'

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
N_BINS = 51
PAD_TOKEN = "<pad>"
PAD_VALUE = N_BINS
MASK_VALUE = N_BINS + 1
MAX_SEQ_LEN = 17007 + 1
cell_emb_style = "cls"
load_model = True

# Multi-run configuration
SEEDS = [100, 101, 102, 103, 104]
FIG_DIR = Path("save_figs")


def set_global_seed(seed: int):
    """Seed all relevant RNGs for reproducibility of a single run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_combine_data(disease_list: list, data_dir: str, rename_map: dict) -> anndata.AnnData:
    """Load multiple cancer datasets (RPPA), tag each with its pancancer type, and concatenate."""
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

        cols_to_keep = [c for c in protein.columns if c in rename_map]
        protein = protein[cols_to_keep]
        protein = protein.loc[common_samples].rename(columns=rename_map)

        adata = anndata.AnnData(X=protein)
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

    return combined_adata


def get_pancancer_embeddings(model: PerformerModel, adata: anndata.AnnData, vocab: GeneVocab) -> np.ndarray:
    """Use the pretrained protein model to produce a sample-level embedding for every row of adata."""
    preprocessor = Preprocessor(
        use_key="X",
        filter_gene_by_counts=False,
        filter_cell_by_counts=False,
        normalize_total=False,
        log1p=False,
        binning=N_BINS,
        result_binned_key="X_binned",
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

    print("Generating embeddings for all samples...")
    src_key_padding_mask = all_gene_ids.eq(vocab[PAD_TOKEN])
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        cell_embeddings = model.encode_batch(
            all_gene_ids,
            all_values.float(),
            src_key_padding_mask=src_key_padding_mask,
            batch_size=32,
            time_step=0,
            return_np=True,
        )

    cell_embeddings = np.array(cell_embeddings)
    norm = np.linalg.norm(cell_embeddings, axis=1, keepdims=True)
    cell_embeddings = cell_embeddings / (norm + 1e-6)

    return cell_embeddings


def run_single_clustering(adata: anndata.AnnData, embeddings: np.ndarray, seed: int,
                          fig_dir: Path) -> dict:
    """Run neighbors + leiden + t-SNE with a given seed, return metrics and save figure."""
    set_global_seed(seed)

    adata_run = adata.copy()
    adata_run.obsm['X_omix'] = embeddings

    print(f"\n=== [seed={seed}] Clustering & Metrics ===")
    sc.pp.neighbors(adata_run, use_rep='X_omix', n_neighbors=30, random_state=seed)
    sc.tl.leiden(adata_run, resolution=1.0, key_added='leiden_clusters', random_state=seed)

    true_labels = adata_run.obs['pancancer_type']
    predicted_labels = adata_run.obs['leiden_clusters']

    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    silhouette = silhouette_score(embeddings, predicted_labels)

    print(f"[seed={seed}] ARI: {ari:.4f}")
    print(f"[seed={seed}] NMI: {nmi:.4f}")
    print(f"[seed={seed}] Silhouette: {silhouette:.4f}")

    print(f"[seed={seed}] Visualizing with t-SNE...")
    sc.tl.tsne(adata_run, use_rep='X_omix', random_state=seed)

    fig = sc.pl.tsne(
        adata_run,
        color='pancancer_type',
        title=f't-SNE of Protein Embeddings by Cancer Type (seed={seed})',
        show=False,
        return_fig=True,
    )
    fig_path = fig_dir / f'embeddings_protein_pretrain{load_model}_seed{seed}.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"[seed={seed}] t-SNE plot saved to {fig_path}")

    return {
        'seed': seed,
        'ari': float(ari),
        'nmi': float(nmi),
        'silhouette': float(silhouette),
    }


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading pre-trained model and vocab...")
    model_dir = Path(LOAD_MODEL_PATH)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "model_e5.pt"
    vocab_file = model_dir / "vocab.json"
    print('loading from', model_file)

    if not all([model_config_file.exists(), model_file.exists(), vocab_file.exists()]):
        raise FileNotFoundError(f"Model files not found in {model_dir}. Please check the path.")

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

    if load_model:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file, map_location=DEVICE)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        for k, v in pretrained_dict.items():
            print(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    print('pretrain:', load_model)
    model.to(DEVICE)
    model.eval()

    with open(RENAME_MAP_PATH, 'r') as f:
        rename_map = json.load(f)
    adata = load_and_combine_data(DISEASE_NAME_LIST, DATA_ROOT_DIR, rename_map)

    # Model forward is deterministic in eval mode, so compute embeddings once
    # and reuse across seeds.
    embeddings = get_pancancer_embeddings(model, adata, vocab)
    print("Embeddings shape:", embeddings.shape)

    results = []
    for seed in SEEDS:
        metrics = run_single_clustering(adata, embeddings, seed, FIG_DIR)
        results.append(metrics)

    results_df = pd.DataFrame(results)
    print("\n================ Per-seed results ================")
    print(results_df.to_string(index=False))

    ari_mean, ari_std = results_df['ari'].mean(), results_df['ari'].std(ddof=1)
    nmi_mean, nmi_std = results_df['nmi'].mean(), results_df['nmi'].std(ddof=1)
    sil_mean, sil_std = results_df['silhouette'].mean(), results_df['silhouette'].std(ddof=1)

    print("\n================ Aggregated (mean ± std) ================")
    print(f"ARI: {ari_mean:.4f} ± {ari_std:.4f}")
    print(f"NMI: {nmi_mean:.4f} ± {nmi_std:.4f}")
    print(f"Silhouette: {sil_mean:.4f} ± {sil_std:.4f}")

    csv_path = FIG_DIR / 'multirun_protein_metrics.csv'
    results_df.to_csv(csv_path, index=False)

    summary_path = FIG_DIR / 'multirun_protein_metrics_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Seeds: {SEEDS}\n")
        f.write(results_df.to_string(index=False) + "\n\n")
        f.write(f"ARI: {ari_mean:.4f} ± {ari_std:.4f}\n")
        f.write(f"NMI: {nmi_mean:.4f} ± {nmi_std:.4f}\n")
        f.write(f"Silhouette: {sil_mean:.4f} ± {sil_std:.4f}\n")

    print(f"\nPer-seed metrics saved to {csv_path}")
    print(f"Summary saved to {summary_path}")


if __name__ == '__main__':
    main()
