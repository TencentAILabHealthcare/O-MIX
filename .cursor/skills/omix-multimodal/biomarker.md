# Biomarker analysis (gradient attribution)

Goal: identify the top-K genes / proteins / CpG sites whose values drive a
specific prediction (typically prognosis or disease label) made by a
fine-tuned O-MIX model. Uses **Layer Integrated Gradients** from `captum`.

Single script: `code/biomarker_analysis/top100_genes_extraction_gradient.py`.

## Prerequisites

This is a **post-finetune** analysis. The user must already have:

1. A pretrained Stage-II checkpoint at `PRETRAIN_DIR` (provides `args.json`
   + vocab files + the encoder architecture).
2. A fine-tuned model directory at `FINETUNE_DIR` containing per-fold
   subdirectories `fold1/`, `fold2/`, ... each with a `best_model.pt`.

If either is missing, point the user back to:
- `embedding.md` / `finetune.md` for pretrain + finetune steps, or
- the project root README for full pretraining.

## Configure the script

Edit the top of `top100_genes_extraction_gradient.py`:

```python
PRETRAIN_DIR = "../pretrain/save/omix_o_pretrain"   # config source
FINETUNE_DIR = "survival_file_dir/survival_multimodal_finetune-performer-BRCA-gateFalse-hvgFalse-DROPOUT0.1"

MODALITIES_ORDER = ['RNA', 'Protein', 'METHYL']
gpu_index = 2
BATCH_SIZE = 1   # gradient attribution is memory-heavy
DROPOUT = 0.1    # must match the finetune run's dropout
```

The `FINETUNE_DIR` naming follows the convention from `finetune.md` §2:
```
survival_file_dir_novel/survival_multimodal_finetune-performer-<CANCER>-gate<bool>-hvg<bool>-DROPOUT<value>/
```
For drug response or classification finetune dirs, adjust the relative path
accordingly.

## Run

```bash
conda activate omix
cd code/biomarker_analysis
PYTHONPATH=../ python top100_genes_extraction_gradient.py
```

The script will, for each fold:

1. Rebuild encoder_dict + fusion_model + the task head exactly as in the
   finetune script.
2. Load `<FINETUNE_DIR>/fold<k>/best_model.pt` into the full model.
3. Wrap the model in `LayerIntegratedGradients` targeting the **input
   embedding layer of each modality**.
4. Compute attributions over the validation samples of that fold.
5. Average absolute attribution per gene / protein / CpG.
6. Rank and save the top-100 per modality.

## Outputs

| File | Where | Content |
|---|---|---|
| `fold_<k>_full_gene_importance.csv` | `FINETUNE_DIR/fold_<k>/` | per-fold attribution table |
| `FINAL_all_genes_aggregated.csv` | `FINETUNE_DIR/` | cross-fold aggregation of all features |
| `FINAL_robust_top100_genes.csv` | `FINETUNE_DIR/` | top-100 features that survive cross-fold aggregation |

The `FINAL_robust_top100_genes.csv` is the one most users want for biological
interpretation — it filters out fold-specific noise.

## Per-fold prerequisites

The script (L454) reads three files from each `fold_<k>/`:

```python
files_needed = ["processed_data.pt", "best_model.pt", "test_idx.npy"]
```

All three are written automatically by `finetune/prognosis_prediction.py`
during training. If the user did the fine-tune via a different script or
moved the fold dirs, they must ensure these three files are present.

## Gotchas

1. **Only the RNA encoder is attributed by default.** L180:
   ```python
   self.target_layer = model.encoder_dict['RNA'].transformer_encoder.net
   ```
   `LayerIntegratedGradients` is hooked **only on RNA**. Protein and
   Methylation get no attributions, even though they participate in the
   forward pass. To get per-protein or per-CpG importance, the user must
   instantiate additional `LayerIntegratedGradients` against
   `encoder_dict['Protein']` / `encoder_dict['METHYL']` and aggregate
   separately.

2. **Encoder weights may be overwritten after fold load.** Around L542 the
   script calls `model.encoder_dict[mod].load_state_dict(pretrained_ckpt['encoder_dict'][mod], strict=True)`
   **after** `best_model.pt` is loaded. If the user does not want the
   pretraining-stage encoder to clobber their fine-tuned encoder, comment
   this block out — otherwise the attributions describe the pretraining
   model, not the fine-tuned one.

3. **`DROPOUT` must match the finetune config exactly.** The script rebuilds
   the model architecture from `args.json` + this single `DROPOUT` knob,
   then loads the fold checkpoint. The shipped default is `0.1`, but
   `prognosis_prediction.py` ships with `0.2`. Override here to whichever
   value the user actually trained with — a mismatch may silently produce
   garbage attributions.

4. **Output directory naming inconsistency.** The fine-tune script writes
   to `survival_file_dir_novel/...` (note the `_novel` suffix), while the
   biomarker script's default example points at `survival_file_dir/...`.
   Set `FINETUNE_DIR` to whichever path the user actually has on disk.

5. **`BATCH_SIZE = 1` is required** in most setups. Integrated gradients
   computes per-sample attributions with internal interpolation; even
   batch 2 typically OOMs on 24 GB cards. Raise only if VRAM allows.

6. **Long runtime.** Each sample requires ~10 forward+backward passes
   (`n_steps=10` default in this script). A 5-fold × ~200-sample BRCA
   survival run takes ~30 minutes on a single A100.

7. **Target output must be scalar.** The script wires up a single scalar
   target (Cox risk score for prognosis, logit for classification). If
   adapting to a multi-class head, specify `target=class_idx` inside the
   `lig.attribute(...)` call.

8. **CpG IDs are not gene names.** If the user extends the script to cover
   Methylation, the resulting CSV rows contain probe-level CpG IDs (e.g.
   `cg00212031`); to convert to gene-level, join against an Illumina
   manifest annotation. The shipped `data/gene_name_mapping/` covers only
   protein names, not CpGs.

## Where to go next

- Plot a heatmap of top-K attributions across folds → see plotting
  conventions in `code/clustering/save_figs/` for the style used elsewhere.
- Validate biological relevance externally → cross-reference the top genes
  against public knowledge bases (e.g. MSigDB pathway enrichment, OMIM,
  DisGeNET) using your tool of choice.
