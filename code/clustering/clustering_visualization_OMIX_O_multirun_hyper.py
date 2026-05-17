# %%
"""
Hyper (multi-checkpoint) OMIX-O (multi-modal) clustering & visualization.

For a single pretrained run, sweep over a list of training-epoch checkpoints
(e.g. epochs 1 / 3 / 5 / 8 / 10) and, for each checkpoint, run the downstream
clustering / visualization pipeline with multiple random seeds. ARI / NMI /
Silhouette are reported as mean ± std per epoch.

Data loading, tokenization, and model construction (3 modality encoders +
OmniFusionBlock) are performed ONCE; only the encoder/fusion ``state_dict`` is
swapped between epochs, and fused embeddings are recomputed from that.
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
from anndata import AnnData
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)
from scipy.sparse import issparse
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.append("../")
from omix.model.omni_fusion import OmniFusionBlock
from omix.model.model_multi import PerformerModel
from omix.preprocess_bulk import Preprocessor as Preprocessor_RNA
from omix.preprocess_rppa import Preprocessor as Preprocessor_Protein
from omix.tokenizer import GeneVocab, tokenize_and_pad_batch

LOAD_MODEL_PATH = "../pretrain/save/omix_o_pretrain"
# Used in the figure filename so multiple runs don't overwrite each other.
USED_DIR_TAG = Path(LOAD_MODEL_PATH).name

DISEASE_NAME_LIST = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC', 'KICH',
                     'KIRC', 'KIRP',  'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'READ', 'SARC', 'SKCM',
                     'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS']

DATA_ROOT_DIR = "../../data/GDAC"

gpu_index = 0
DEVICE = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(gpu_index)
BATCH_SIZE = 4
N_BINS = 51
PAD_TOKEN = "<pad>"
PAD_VALUE = N_BINS
MASK_VALUE = N_BINS + 1
max_seq_len_protein = 17007 + 1
max_seq_len_rna = 19205 + 1
max_seq_len_methyl = 12920 + 1
RENAME_MAP_PATH = '../../data/gene_name_mapping/protein_name_mapping.json'
load_model = True
cell_emb_style = "cls"
dropout = 0.0

# Multi-run configuration
SEEDS = [100, 101, 102, 103, 104]
EPOCHS_TO_TEST = [1, 3, 5, 8, 10]
FIG_DIR = Path("save_figs")


def set_global_seed(seed: int):
    """Seed all relevant RNGs for reproducibility of a single run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_combine_data(disease_list: list, data_dir: str, rename_map: dict):
    """Load RNA / Protein / Methylation data per disease, concat, and return three AnnData objects."""
    used_gene_file = '../../data/GDAC/filtered_genes_list.csv'
    gene_list_df = pd.read_csv(used_gene_file)
    genes_to_keep = gene_list_df['gene_name'].tolist()

    used_gene_file_methyl = '../../data/GDAC/filtered_genes_list_methyl.csv'
    gene_list_df_methyl = pd.read_csv(used_gene_file_methyl)
    genes_to_keep_methyl = gene_list_df_methyl['gene_name'].tolist()

    adata_list = []
    adata_protein_list = []
    adata_methyl_list = []
    for disease in disease_list:
        print(f"Loading data for {disease}...")
        disease_path = os.path.join(data_dir, disease)
        if not os.path.exists(disease_path):
            raise FileNotFoundError(f"Data directory for {disease} not found at {disease_path}")

        labels = pd.read_csv(os.path.join(disease_path, 'labels.csv'), sep='\t', index_col=0)
        mrna = pd.read_csv(os.path.join(disease_path, 'mRNA_TPM.csv'), index_col=0)
        methyl = pd.read_csv(os.path.join(disease_path, 'methylation.csv'), index_col=0)
        protein = pd.read_csv(os.path.join(disease_path, 'RPPA.csv'), index_col=0)

        genes_present_in_file = mrna.columns
        common_genes = [gene for gene in genes_to_keep if gene in genes_present_in_file]

        genes_present_in_file_methyl = methyl.columns
        common_genes_methyl = [gene for gene in genes_to_keep_methyl if gene in genes_present_in_file_methyl]

        mrna.index = mrna.index.str.slice(0, 12).str.lower()
        mrna = mrna[~mrna.index.duplicated(keep='first')]

        protein.index = protein.index.str.slice(0, 12).str.lower()
        protein = protein[~protein.index.duplicated(keep='first')]

        methyl.index = methyl.index.str.slice(0, 12).str.lower()
        methyl = methyl[~methyl.index.duplicated(keep='first')]

        common_samples = mrna.index.intersection(protein.index).intersection(methyl.index)

        mrna = mrna.loc[common_samples]
        methyl = methyl.loc[common_samples]

        cols_to_keep = [c for c in protein.columns if c in rename_map]
        protein = protein[cols_to_keep]
        protein = protein.loc[common_samples].rename(columns=rename_map)

        disease_name = 'COADREAD' if disease in ('COAD', 'READ') else disease

        adata = AnnData(X=mrna)
        adata.var_names_make_unique()
        adata.var["gene_name"] = adata.var.index.tolist()
        adata.obs['pancancer_type'] = disease_name
        adata.X = np.nan_to_num(adata.X, nan=0.0)

        adata_protein = AnnData(X=protein)
        adata_protein.var_names_make_unique()
        adata_protein.var["gene_name"] = adata_protein.var.index.tolist()
        adata_protein.obs['pancancer_type'] = disease_name

        adata_methyl = AnnData(X=methyl)
        adata_methyl.var_names_make_unique()
        adata_methyl.var["gene_name"] = adata_methyl.var.index.tolist()
        adata_methyl.obs['pancancer_type'] = disease_name

        adata_list.append(adata)
        adata_methyl_list.append(adata_methyl)
        adata_protein_list.append(adata_protein)

    combined_adata = anndata.concat(adata_list, join='outer', label='batch', index_unique=None)
    combined_adata.obs_names_make_unique()
    combined_adata.X = np.nan_to_num(combined_adata.X, nan=0.0)
    combined_adata_protein = anndata.concat(adata_protein_list, join='outer', label='batch', index_unique=None)
    combined_adata_protein.obs_names_make_unique()
    combined_adata_methyl = anndata.concat(adata_methyl_list, join='outer', label='batch', index_unique=None)
    combined_adata_methyl.obs_names_make_unique()
    return combined_adata, combined_adata_protein, combined_adata_methyl


def get_pancancer_embeddings_flexible(fusion_model, encoder_dict, data_dict, vocab_mod) -> np.ndarray:
    """Generate fused multi-modal embeddings by running each encoder then the fusion block."""
    fusion_model.eval()
    for encoder in encoder_dict.values():
        encoder.eval()

    device = next(fusion_model.parameters()).device
    print(f"Model is on device: {device}")

    modalities = list(data_dict.keys())
    tensors = [tensor for modality in modalities for tensor in data_dict[modality]]

    full_dataset = TensorDataset(*tensors)
    test_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False)

    cell_embeddings = []
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Generating Embeddings"):
            fusion_input_cls = []
            mod_types_list = []

            batch_size = batch_data[0].shape[0]
            num_tensors_per_modality = 3

            for i, modality in enumerate(modalities):
                start_idx = i * num_tensors_per_modality
                ids, values, padding_mask = batch_data[start_idx: start_idx + num_tensors_per_modality]

                ids, values, padding_mask = ids.to(device), values.to(device), padding_mask.to(device)

                encoded_output, _ = encoder_dict[modality](
                    src=ids,
                    values=values,
                    src_key_padding_mask=padding_mask,
                    return_seq_embedding=True,
                )

                if hasattr(fusion_model, "module"):
                    encoded_samples = fusion_model.module.run_adapter(modality, encoded_output)
                else:
                    encoded_samples = fusion_model.run_adapter(modality, encoded_output)

                cls_token = encoded_samples[:, 0, :].unsqueeze(1).clone()
                fusion_input_cls.append(cls_token)

                mod_id = vocab_mod[modality]
                current_mod_types = torch.full((batch_size, 1), mod_id, dtype=torch.long, device=device)
                mod_types_list.append(current_mod_types)

            mod_types = torch.cat(mod_types_list, dim=1)
            batch_mcs = torch.tensor([0] * batch_data[0].shape[0]).to(device)

            cell_embeddings_batch = fusion_model(
                *fusion_input_cls,
                expert_indices=batch_mcs,
                mod_types=mod_types,
                return_cell_embedding=True,
            )
            cell_embeddings.append(cell_embeddings_batch.cpu().numpy())

    cell_embeddings = np.concatenate(cell_embeddings, axis=0)

    norm = np.linalg.norm(cell_embeddings, axis=1, keepdims=True)
    cell_embeddings = cell_embeddings / (norm + 1e-6)

    return cell_embeddings


def prepare_data_and_models():
    """Load vocabs, build encoder_dict + fusion_model, preprocess + tokenize data ONCE.

    Returns everything that can be reused across checkpoints. Per-epoch we will only
    swap the model state_dicts and re-run ``get_pancancer_embeddings_flexible``.
    """
    print("Loading model config and vocabs...")
    model_dir = Path(LOAD_MODEL_PATH)
    model_config_file = model_dir / "args.json"
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)

    if "modality_dict" in model_configs:
        vocab_mod = {k: v for k, v in model_configs["modality_dict"].items()}
        if PAD_TOKEN not in vocab_mod:
            vocab_mod[PAD_TOKEN] = len(vocab_mod)
    else:
        raise KeyError("modality_dict not found in args.json")

    special_tokens = [PAD_TOKEN, "<cls>", "<eoc>"]

    vocab_RNA = GeneVocab.from_file(model_dir / "vocab_rna.json")
    for s in special_tokens:
        if s not in vocab_RNA:
            vocab_RNA.append_token(s)

    vocab_Protein = GeneVocab.from_file(model_dir / "vocab_protein.json")
    for s in special_tokens:
        if s not in vocab_Protein:
            vocab_Protein.append_token(s)

    vocab_methyl = GeneVocab.from_file(model_dir / "vocab_methyl.json")
    for s in special_tokens:
        if s not in vocab_methyl:
            vocab_methyl.append_token(s)

    with open(RENAME_MAP_PATH, 'r') as f:
        rename_map = json.load(f)

    adata, adata_protein, adata_methyl = load_and_combine_data(
        DISEASE_NAME_LIST, DATA_ROOT_DIR, rename_map
    )

    preprocessor_rna = Preprocessor_RNA(
        use_key="X",
        filter_gene_by_counts=False,
        filter_cell_by_counts=False,
        normalize_total=False,
        log1p=True,
        binning=51,
        result_binned_key="X_binned",
        repair=False,
    )
    preprocessor_rna(adata)

    preprocessor_protein = Preprocessor_Protein(
        use_key="X",
        filter_gene_by_counts=False,
        filter_cell_by_counts=False,
        normalize_total=False,
        result_normed_key="X_normed",
        log1p=False,
        result_log1p_key="X_log1p",
        subset_hvg=False,
        hvg_flavor="seurat",
        binning=51,
        result_binned_key="X_binned",
    )
    preprocessor_protein(adata_protein)

    preprocessor_methyl = Preprocessor_RNA(
        use_key="X",
        filter_gene_by_counts=False,
        filter_cell_by_counts=False,
        normalize_total=False,
        log1p=False,
        binning=51,
        result_binned_key="X_binned",
        repair=False,
    )
    preprocessor_methyl(adata_methyl)

    gene_ids_RNA = np.array(vocab_RNA(adata.var_names.tolist()), dtype=int)
    proteins_ids_Protein = np.array(vocab_Protein(adata_protein.var_names.tolist()), dtype=int)
    gene_ids_methyl = np.array(vocab_methyl(adata_methyl.var_names.tolist()), dtype=int)

    input_layer_key = "X_binned"
    RNA_data = (
        adata.layers[input_layer_key].toarray()
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )
    Protein_data = (
        adata_protein.layers[input_layer_key].toarray()
        if issparse(adata_protein.layers[input_layer_key])
        else adata_protein.layers[input_layer_key]
    )
    methyl_data = (
        adata_methyl.layers[input_layer_key].toarray()
        if issparse(adata_methyl.layers[input_layer_key])
        else adata_methyl.layers[input_layer_key]
    )

    tokenized_RNA = tokenize_and_pad_batch(
        RNA_data, gene_ids_RNA, max_len=max_seq_len_rna, vocab=vocab_RNA,
        pad_token=PAD_TOKEN, pad_value=PAD_VALUE, append_cls=True, include_zero_gene=False,
    )
    tokenized_Protein = tokenize_and_pad_batch(
        Protein_data, proteins_ids_Protein, max_len=max_seq_len_protein, vocab=vocab_Protein,
        pad_token=PAD_TOKEN, pad_value=PAD_VALUE, append_cls=True, include_zero_gene=False,
    )
    tokenized_methyl = tokenize_and_pad_batch(
        methyl_data, gene_ids_methyl, max_len=max_seq_len_methyl, vocab=vocab_methyl,
        pad_token=PAD_TOKEN, pad_value=PAD_VALUE, append_cls=True, include_zero_gene=False,
    )

    all_gene_ids_rna = tokenized_RNA['genes']
    all_values_rna = tokenized_RNA['values']
    src_key_padding_mask_rna = all_gene_ids_rna.eq(vocab_RNA[PAD_TOKEN])

    all_gene_ids_protein = tokenized_Protein['genes']
    all_values_protein = tokenized_Protein['values']
    src_key_padding_mask_protein = all_gene_ids_protein.eq(vocab_Protein[PAD_TOKEN])

    all_gene_ids_methyl = tokenized_methyl['genes']
    all_values_methyl = tokenized_methyl['values']
    src_key_padding_mask_methyl = all_gene_ids_methyl.eq(vocab_methyl[PAD_TOKEN])

    encoder_dict = {}
    encoder_dict['RNA'] = PerformerModel(
        ntoken=len(vocab_RNA),
        d_model=model_configs["layer_size"],
        nhead=model_configs["nhead"],
        d_hid=model_configs["layer_size"],
        nlayers=model_configs["nlayers"],
        vocab=vocab_RNA,
        pad_token=model_configs["pad_token"],
        pad_value=model_configs["pad_value"],
        n_input_bins=model_configs["n_bins"] + 2,
        cell_emb_style="cls",
        input_emb_style="category",
    )
    encoder_dict['Protein'] = PerformerModel(
        ntoken=len(vocab_Protein),
        d_model=model_configs["layer_size"],
        nhead=model_configs["nhead"],
        d_hid=model_configs["layer_size"],
        nlayers=model_configs["nlayers"],
        vocab=vocab_Protein,
        pad_token=model_configs["pad_token"],
        pad_value=model_configs["pad_value"],
        n_input_bins=model_configs["n_bins"] + 2,
        cell_emb_style="cls",
        input_emb_style="category",
    )
    encoder_dict['METHYL'] = PerformerModel(
        ntoken=len(vocab_methyl),
        d_model=model_configs["layer_size"],
        nhead=model_configs["nhead"],
        d_hid=model_configs["layer_size"],
        nlayers=model_configs["nlayers"],
        vocab=vocab_methyl,
        pad_token=model_configs["pad_token"],
        pad_value=model_configs["pad_value"],
        n_input_bins=model_configs["n_bins"] + 2,
        cell_emb_style="cls",
        input_emb_style="category",
    )
    print(f'dropout: {dropout}')
    print(f'cell_emb_style: {cell_emb_style}')

    full_modality_index = 0
    total_length = 3
    fusion_model = OmniFusionBlock(
        model_configs["num_modalities"], full_modality_index, total_length,
        model_configs["layer_size"], model_configs["num_layers_fus"], model_configs["num_experts"],
        model_configs["num_routers"], model_configs["top_k"], model_configs["nhead"],
        model_configs["dropout"], vocab_mod,
    )

    for key in encoder_dict:
        encoder_dict[key] = encoder_dict[key].to(DEVICE)
    fusion_model = fusion_model.to(DEVICE)

    data_dict = {
        'RNA': (all_gene_ids_rna, all_values_rna, src_key_padding_mask_rna),
        'Protein': (all_gene_ids_protein, all_values_protein, src_key_padding_mask_protein),
        'METHYL': (all_gene_ids_methyl, all_values_methyl, src_key_padding_mask_methyl),
    }

    return adata, encoder_dict, fusion_model, data_dict, vocab_mod


def load_checkpoint_into_models(encoder_dict, fusion_model, ckpt_path: Path):
    """Replace encoder + fusion weights with those from ``ckpt_path`` (shape-matching keys only)."""
    print(f"Loading checkpoint from {ckpt_path}")
    pretrained_state = torch.load(ckpt_path, map_location='cpu')

    for modality_name in ('RNA', 'Protein', 'METHYL'):
        model_dict_mod = encoder_dict[modality_name].state_dict()
        pretrained_dict_mod = {
            k: v
            for k, v in pretrained_state['encoder_dict'][modality_name].items()
            if k in model_dict_mod and v.shape == model_dict_mod[k].shape
        }
        print(f"  [{modality_name}] matched {len(pretrained_dict_mod)} / {len(model_dict_mod)} tensors.")
        model_dict_mod.update(pretrained_dict_mod)
        encoder_dict[modality_name].load_state_dict(model_dict_mod)
        encoder_dict[modality_name].eval()

    model_dict_fusion = fusion_model.state_dict()
    pretrained_dict_fusion = {
        k: v
        for k, v in pretrained_state['fusion_model'].items()
        if k in model_dict_fusion and v.shape == model_dict_fusion[k].shape
    }
    print(f"  [fusion] matched {len(pretrained_dict_fusion)} / {len(model_dict_fusion)} tensors.")
    model_dict_fusion.update(pretrained_dict_fusion)
    fusion_model.load_state_dict(model_dict_fusion)
    fusion_model.eval()


def run_single_clustering(adata: anndata.AnnData, embeddings: np.ndarray, seed: int,
                          epoch: int, fig_dir: Path) -> dict:
    """Run neighbors + leiden + t-SNE with a given seed, return metrics and save figure."""
    set_global_seed(seed)

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
        title=f't-SNE of OMIX-O Embeddings by Cancer Type (epoch={epoch}, seed={seed})',
        show=False,
        return_fig=True,
    )
    fig_path = fig_dir / f'embeddings_multimodal_pretrain{load_model}_{USED_DIR_TAG}_e{epoch}_seed{seed}.png'
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

    # Validate every requested checkpoint up-front so we fail fast.
    model_dir = Path(LOAD_MODEL_PATH)
    ckpt_files = {epoch: model_dir / f"model_e{epoch}.pt" for epoch in EPOCHS_TO_TEST}
    missing = [str(p) for p in ckpt_files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")

    # Seed once for model construction / data loading; per-run seeds are set inside the loop.
    set_global_seed(SEEDS[0])

    # Build models + tokenize data once; reused for every checkpoint.
    adata, encoder_dict, fusion_model, data_dict, vocab_mod = prepare_data_and_models()

    all_results = []
    for epoch in EPOCHS_TO_TEST:
        print(f"\n################ Epoch {epoch} ################")
        if load_model:
            load_checkpoint_into_models(encoder_dict, fusion_model, ckpt_files[epoch])
        else:
            # Still need eval mode for deterministic forward.
            for enc in encoder_dict.values():
                enc.eval()
            fusion_model.eval()

        embeddings = get_pancancer_embeddings_flexible(
            fusion_model, encoder_dict, data_dict, vocab_mod,
        )
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

    csv_path = FIG_DIR / 'multirun_omix_o_hyper_metrics.csv'
    results_df.to_csv(csv_path, index=False)

    summary_csv_path = FIG_DIR / 'multirun_omix_o_hyper_metrics_summary.csv'
    summary_df.to_csv(summary_csv_path, index=False)

    summary_txt_path = FIG_DIR / 'multirun_omix_o_hyper_metrics_summary.txt'
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
