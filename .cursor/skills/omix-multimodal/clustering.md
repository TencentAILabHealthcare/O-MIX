# Clustering and visualization

Goal: take pretrained O-MIX embeddings (or raw omics matrices) and visualize
how well they separate pan-cancer types. The output is a t-SNE / UMAP plot
plus three quality metrics: **ARI, NMI, Silhouette**.

Scripts live in `code/clustering/`. Twelve scripts in total, organized as a
matrix of {modality} × {run variant}:

| Modality | Single run | Multi-seed average | Hyperparameter grid |
|---|---|---|---|
| RNA | `clustering_visualization_RNA.py` | `_multirun.py` | `_multirun_hyper.py` |
| Protein | `clustering_visualization_Protein.py` | `_multirun.py` | `_multirun_hyper.py` |
| Methylation | `clustering_visualization_methylation.py` | `_multirun.py` | `_multirun_hyper.py` |
| O-MIX-O (fused) | `clustering_visualization_OMIX_O.py` | `_multirun.py` (5 seeds) | `_multirun_hyper.py` (5 epochs × seeds) |

There is no `OMIX_T` clustering script — text-only modality is not a
clustering target on its own.

## Decision: which variant to run

- **Just want one figure for a slide / quick check?** → use the plain
  `clustering_visualization_<mod>.py`.
- **Reporting numbers in a paper?** → use `_multirun.py`, which averages ARI
  / NMI / Silhouette over multiple random seeds (typically 5) and writes a
  CSV row per seed.
- **Comparing across pretraining epochs?** → use `_multirun_hyper.py`,
  which sweeps `EPOCHS_TO_TEST = [1, 3, 5, 8, 10]` (a list of checkpoint
  epochs, **not** Leiden resolution / `n_neighbors`) crossed with the
  random-seed list, and emits one CSV row per (epoch, seed). To actually
  sweep Leiden hyperparameters, the user must edit the loop themselves.

## Common knobs (any script)

Open the top of any clustering script — they all share the same config block:

```python
LOAD_MODEL_PATH = "../pretrain/save/omix_o_pretrain"
DATA_ROOT_DIR   = "../../data/GDAC"
DISEASE_NAME_LIST = ['ACC', 'BLCA', 'BRCA', ...]   # 30 TCGA types
gpu_index = 0
BATCH_SIZE = 4
cell_emb_style = "cls"
```

Override before launching if the user has different paths or only wants
a subset of cancer types.

## Launch

```bash
conda activate omix
cd code/clustering

PYTHONPATH=../ python clustering_visualization_OMIX_O.py

PYTHONPATH=../ python clustering_visualization_OMIX_O_multirun.py

PYTHONPATH=../ python clustering_visualization_OMIX_O_multirun_hyper.py
```

The script will:

1. Load the Stage-II checkpoint and per-modality vocabs.
2. Tokenize and encode all samples from `DATA_ROOT_DIR/<cancer>/`.
3. Compute neighbors + Leiden clusters with `scanpy`.
4. Compute ARI / NMI / Silhouette against the true `pancancer_type` label.
5. Save a t-SNE PNG and print the metrics.

## What the script saves

- Single run: `tsne_omix_o_by_cancer_type.png`, `tsne_omix_o_by_leiden.png`,
  and a stdout summary of the three metrics.
- Multi-seed: same plots for the last seed, plus `metrics_seeds.csv` with
  one row per seed.
- Hyper grid: `metrics_hyper.csv` with one row per (resolution, n_neighbors)
  cell.

Outputs are written into `code/clustering/save_figs/` by default.

## Comparison baseline (PCA + Leiden on raw data)

Each script **defines** a helper `run_raw_data_analysis(adata_raw)` that
clusters the raw RNA / Protein / Methyl matrix via PCA + Leiden (no O-MIX
embedding). It is **not** called from `main()` by default — the function
sits in the script as an opt-in. To produce the apples-to-apples comparison
"O-MIX embedding vs. classical PCA":

1. Load the per-modality `AnnData` you want to compare.
2. Make a copy: `adata_raw = adata.copy()`.
3. Add `adata_raw.obs['pancancer_type']` if not already present.
4. Call `run_raw_data_analysis(adata_raw)` manually before or after the
   O-MIX section.

The function prints `ARI / NMI / silhouette` for the raw baseline and saves
a `tsne_raw_by_cancer_type_*.png` plot.

## Gotchas

1. **t-SNE not UMAP by default.** The shipped scripts use `sc.tl.tsne`. To
   switch to UMAP, replace with `sc.tl.umap` and `sc.pl.umap`. Metrics are
   identical because they are computed on the neighbor graph, not the 2D
   embedding.
2. **Silhouette can be expensive** on > 5,000 samples (O(n²)). If the user
   sees the script hang at the "computing silhouette" step, they should
   either subsample or set `silhouette_score(..., sample_size=2000)`.
3. **`pancancer_type` label must exist in `adata.obs`.** The TCGA `.h5ad`
   files in `data/GDAC/<cancer>/` set this automatically. If a user supplies
   their own AnnData they must add the label column before launching.
4. **GPU memory is the bottleneck**, not the clustering step. Encoding 30
   cancers × ~500 samples × 3 modalities at `BATCH_SIZE=4` fits on a single
   24 GB card; lower batch size for smaller cards.

## Linking back

Once a user has a clean cluster plot, they typically move to one of:
- `retrieval.md` → "now check whether nearest neighbors are the same cancer"
- `biomarker.md` → "now extract the genes that drive the separation"
- `finetune.md` § 1 → "now actually train a classifier on top"
