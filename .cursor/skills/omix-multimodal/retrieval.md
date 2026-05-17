# Cross-modal retrieval and zero-shot classification

Two evaluation pipelines live under `code/crossmodal_retrieval/`. Both assume
a Stage-II O-MIX checkpoint is already trained (typically `omix_t_pretrain/`).

## Pipeline overview

| Pipeline | Goal | Scripts | Checkpoint |
|---|---|---|---|
| **Pretraining-validation retrieval** | Quantify cross-modal alignment on the pretraining held-out split (omics ↔ omics or omics ↔ text). | `pretraining_validation_retrival_stage1_save_embedding.py` → `pretraining_validation_retrival_stage2_evaluation.py` | `omix_o_pretrain` (RPM) or `omix_t_pretrain` (RPMT) |
| **Human-disease zero-shot** | Given a clinical query about a disease, retrieve the most similar patient and predict the disease label without training. | `humandisease_retrieval_evaluation.py`, `humandisease_zeroshot_classification.py` | `omix_t_pretrain` **required** (needs text encoder) |
| **OMIM ↔ GSVA baseline** | Compare the learned embedding's disease–gene alignment to a classical GSVA baseline on OMIM. | `omim_save_embedding.py` → `omim_gsva_comparison.py` | `omix_t_pretrain` |

## §1 Pretraining-validation retrieval

**Stage 1 — save embeddings** writes per-modality CLS embeddings + fused
embeddings + sample IDs to disk so Stage 2 can evaluate without re-running
the model.

```bash
cd code/crossmodal_retrieval
PYTHONPATH=../ python pretraining_validation_retrival_stage1_save_embedding.py
```

Key knobs at the top of the script:
- `model_dir` — points at one of `../pretrain/save/omix_{o,t}_pretrain/`.
- `model_epoch` — epoch number (e.g. `8` for `omix_o`, `5` for `omix_t`).
- `valid_ids_path` — JSON listing sample IDs to embed (the held-out
  validation split saved by pretraining as `pretraining_dataset_split.json`).
- Output: a `.h5` or `.npz` per modality plus a fused `.npz`, written next
  to the checkpoint.

**Stage 2 — evaluate** computes top-K retrieval precision/recall and ranking
metrics between modality pairs.

```bash
PYTHONPATH=../ python pretraining_validation_retrival_stage2_evaluation.py
```

Primary outputs: Recall@K and MRR per modality pair, printed and saved as
CSV / JSON in the same folder. Look at the script's `EVAL_PAIRS` list to add
or remove pairs (e.g. `('RNA', 'TEXT')`).

## §2 Human-disease zero-shot retrieval & classification

There are **three** different scripts here, with subtly different inputs and
outputs. Pick the one that matches the user's actual question.

### §2a — `omim_save_embedding.py` (text → omics retrieval)

Encodes OMIM disease names through O-MIX-T's text encoder, then computes a
similarity matrix against all stored patient omics embeddings.

**Important**: the text input is a **hardcoded template**, not a free-form
OMIM description. L361–362 of `omim_save_embedding.py`:

```python
disease_names = list(omim_gene_sets.keys())   # from OMIM_gene_score.json
for d in disease_names:
    p = f"- Patient diagnosed with {d}."
```

So the model sees `"- Patient diagnosed with breast cancer."`, not the
full OMIM text entry. If the user wants long descriptions, they must edit
the prompt construction loop themselves.

Outputs:
- `sim_matrix.csv` — diseases × samples CSLS similarity table.
- `adata_text_emb.h5ad` — cached text embeddings keyed by `disease_names`.

Per-sample predicted disease = `argmax` along the rows.

Required: `omix_t_pretrain` checkpoint + `omix/Youtu_embedding/`.
Input data: `data/cellwhisper/human_disease/OMIM_gene_score.json`.

### §2b — `humandisease_zeroshot_classification.py` (RNA → class-prompt)

Different direction from §2a. Encodes **RNA embeddings of patients** against
a set of **class-name prompts**, and reports classification metrics.

- Reads `adata.obs[<label_col>]` (e.g. disease type) as the ground truth.
- Builds one prompt per class, encodes via the text encoder.
- For each patient, picks the class whose prompt is most similar.

Metrics printed: **accuracy, macro AUROC**. Outputs: similarity CSV plus an
annotated `.h5ad` with predicted labels.

Note: this script is RNA-only — it does not consume Protein / Methyl.

### §2c — `humandisease_retrieval_evaluation.py` (narratives + K-means prototypes)

Uses pre-generated **clinical narrative JSON** (one per patient) plus
K-means class prototypes to compute **Recall@K** retrieval metrics. Not the
right script for "OMIM description as query"; reserve it for narrative-based
retrieval benchmarks.

### How to launch any of the three

```bash
cd code/crossmodal_retrieval
PYTHONPATH=../ python omim_save_embedding.py            # §2a, must run first if §2b/§2c reuse its outputs
PYTHONPATH=../ python humandisease_zeroshot_classification.py
PYTHONPATH=../ python humandisease_retrieval_evaluation.py
```

All three require: `omix_t_pretrain` + `omix/Youtu_embedding/` + `peft`.
**`peft` is referenced by the project but is NOT pinned in
`requirements.txt`** — install separately if missing.

## §3 OMIM vs GSVA baseline

Used in the manuscript to demonstrate that the O-MIX-learned embedding
beats classical GSVA-style enrichment on disease–gene retrieval.

```bash
cd code/crossmodal_retrieval
PYTHONPATH=../ python omim_save_embedding.py   # if not already done
PYTHONPATH=../ python omim_gsva_comparison.py
```

`omim_gsva_comparison.py` expects `gsva_scores_*.csv` to already exist
(produced by an external R / GSVA pipeline). If the user does not have those
files, they should either run the upstream GSVA pipeline first, or skip §3.

## Gotchas

1. **`encoder_dict['TEXT']` requires `omix/Youtu_embedding/`** to be present
   on disk. If the user only has the bare `omix_t_pretrain/` checkpoint
   without the Youtu submodule, `_build_models()` will fail loudly. Fall
   back to `_O` checkpoint + omics-only retrieval in that case.
2. **Stage 1 writes embeddings to the checkpoint folder by default**. On
   a read-only fileshare, change `OUTPUT_DIR` in the script before launching.
3. **PEFT key mismatches** between the saved checkpoint and the freshly
   built LoRA modules — reuse `_fix_peft_keys(...)` from
   `pretraining_validation_retrival_stage1_save_embedding.py` (already
   handles `original_X ↔ original_module.X` swap).
4. **Sample-ID alignment** — the retrieval scripts filter by an
   "intersection" between the valid-ID set and what each modality `.h5ad`
   actually contains. If a modality file is missing some IDs the script
   silently drops them; check the printed `Intersection: N samples` line
   and confirm it matches the user's expectation.

## When the user only wants embeddings (not retrieval metrics)

Redirect them to `embedding.md`. The Stage 1 script in this file is a
heavier-weight version of the same template, with file IO and PEFT handling
added. For a quick "give me one patient's embedding" use case, the simpler
template in `embedding.md` is preferable.
