---
name: omix-multimodal
description: >-
  Guides users through the O-MIX multi-omics foundation model: extracting
  patient embeddings, fine-tuning downstream predictors, cross-modal retrieval
  and zero-shot classification, clustering and visualization, gradient-based
  biomarker analysis, and Stage I/II pretraining from scratch. Use when the
  user mentions O-MIX, OMIX, OmniFusionBlock, multi-omics, RNA + Protein +
  Methylation, RPPA, TCGA cancer classification, prognosis prediction with
  Cox loss, drug response on CCLE, MoE fusion, loading model_e8.pt /
  model_e5.pt, cross-modal retrieval, zero-shot disease classification, OMIM,
  GSVA, t-SNE / UMAP / Leiden clustering, ARI / NMI / silhouette,
  Integrated Gradients / captum biomarker attribution, top-100 genes,
  pretrain_RNA / pretrain_OMIX_O / pretrain_OMIX_T, DDP, or torchrun.
---

# O-MIX multimodal Skill

This Skill operates inside the `OMIX_codebuddy/` project. It assumes the agent
is helping a user with any of seven workflows: **embedding extraction**,
**three downstream fine-tunes** (disease / prognosis / drug response),
**cross-modal retrieval and zero-shot classification**, **clustering and
visualization**, **gradient-based biomarker analysis**, and **Stage I/II
pretraining**.

## Repository orientation

All paths below are relative to the project root `OMIX_codebuddy/`.

> **Important:** the pretrained checkpoints (`code/pretrain/save/*`), the
> text encoder (`code/omix/Youtu_embedding/`), and **downstream evaluation
> cohorts** (`data/GDAC/`, `data/CCLE2019/`, `data/cellwhisper/`) are
> **not** included in the GitHub repository. Download the matching rows
> from `README.md` (§ "Required external resources", tables A and B #8–#10)
> and place each bundle at the path shown there.
>
> The Stage I/II **pretraining corpus** (`data/pretraining_data/` by
> default in several scripts) is **not** distributed with this repo. For
> `pretrain_*.py` or pretraining-validation retrieval, point users to the
> manuscript (*Methods* / *Data availability*) or their own prepared
> AnnData / JSON paths — do **not** suggest a public download bundle for
> `pretraining_data`.

| Asset | Path |
|---|---|
| Core library (`import omix.*`) | `code/omix/` |
| Pretrained Stage-II checkpoint (omics-only) | `code/pretrain/save/omix_o_pretrain/model_e8.pt` |
| Pretrained Stage-II checkpoint (with text) | `code/pretrain/save/omix_t_pretrain/model_e5.pt` |
| Per-modality vocabs | `code/pretrain/save/omix_o_pretrain/vocab_{rna,protein,methyl}.json` |
| Training args snapshot (use for inference) | `code/pretrain/save/omix_o_pretrain/args.json` |
| Downstream scripts | `code/finetune/{disease_classification,prognosis_prediction,drug_response_prediction}.py` |
| TCGA per-cancer data | `data/GDAC/<CANCER>/` (e.g. `BRCA`, `LGG`, `OV`) |
| CCLE drug-response data | `data/CCLE2019/preprocessed/` |
| Protein name mapping | `data/gene_name_mapping/protein_name_mapping.json` |

## Environment

The project uses two conda environments. For the workflows this Skill
covers, only `omix` is needed (the `vllm` env is only required for clinical
narrative generation, which is out of scope here).

```bash
conda create -n omix python=3.10 && conda activate omix
pip install -r requirements.txt

# fastmoe has no PyPI wheel — build from source. For the minimal (non-distributed) build:
git clone https://github.com/laekov/fastmoe.git && cd fastmoe && USE_NCCL=0 python setup.py install
```

`fastmoe` requires a working CUDA toolchain and **must be installed before
the OmniFusionBlock import resolves**. If a user sees `ModuleNotFoundError:
fmoe`, this is the cause. Full options (distributed build, NCCL setup, CUDA
version pinning) are documented in the official guide:
<https://github.com/laekov/fastmoe/blob/master/doc/installation-guide.md>

## Universal invocation rule

Every script lives under a subfolder of `code/` and imports `omix.*` from its
sibling. The agent **must** instruct users to:

1. `cd` into the script's directory (`code/finetune/`, `code/pretrain/`, etc.)
2. Launch python with `PYTHONPATH=../` so `import omix` resolves.

Skipping step 2 produces `ModuleNotFoundError: No module named 'omix'`.

## Decision tree

Pick the workflow that matches the user's intent:

- **"I want patient / sample embeddings from the pretrained model"**
  → follow [embedding.md](embedding.md)

- **"I want to fine-tune O-MIX for pan-cancer disease classification (30 TCGA types)"**
  → follow [finetune.md](finetune.md) §1

- **"I want survival / prognosis prediction with Cox loss"**
  → follow [finetune.md](finetune.md) §2

- **"I want drug response prediction on CCLE"**
  → follow [finetune.md](finetune.md) §3

- **"I want cross-modal retrieval, zero-shot disease classification, or
  OMIM ↔ GSVA comparison"**
  → follow [retrieval.md](retrieval.md)

- **"I want to cluster / visualize the embeddings (t-SNE, UMAP, Leiden,
  ARI / NMI / silhouette)"**
  → follow [clustering.md](clustering.md)

- **"I want to identify the top genes / proteins / CpGs driving a fine-tuned
  prediction (biomarker / gradient attribution)"**
  → follow [biomarker.md](biomarker.md)

- **"I want to retrain O-MIX from scratch (Stage I and/or Stage II)"**
  → follow [pretraining.md](pretraining.md)

- **Anything else** (e.g. clinical-narrative generation via the `vllm` env,
  custom data preprocessing): not covered by this Skill; point the user at
  `README.md` in the project root.

## Key conventions the agent must respect

These come from the actual code in `code/finetune/*.py` and
`code/crossmodal_retrieval/*.py`. Misuse causes silent shape mismatches or
`KeyError`s on checkpoint loading.

1. **Modality character codes.** `MODALITY_CHARS = {'RNA': 'R', 'Protein':
   'P', 'METHYL': 'M', 'TEXT': 'T'}`. The combination string `'RPM'` means
   omics-only; `'RPMT'` includes clinical text. `MODALITIES_TO_KEEP` is the
   list version of the same selection.

2. **Sequence lengths are fixed by the pretrained vocab** and must match
   exactly:
   - `MAX_SEQ_LEN_RNA = 19206`
   - `MAX_SEQ_LEN_PROTEIN = 17008`
   - `MAX_SEQ_LEN_METHYL = 12921`
   - `MAX_SEQ_LEN_TEXT = 1024` (only for `_T` checkpoint)

3. **Special token padding.** After `GeneVocab.from_file(...)`, append
   `<pad>`, `<cls>`, `<eoc>` if absent. Existing scripts do this with:
   ```python
   for s in [args.pad_token, "<cls>", "<eoc>"]:
       if s not in vocab:
           vocab.append_token(s)
   ```

4. **Checkpoint structure.** `torch.load(model_e8.pt)` returns a dict with:
   - `checkpoint['encoder_dict'][mod]` — per-modality `PerformerModel`
     state-dict (`mod` ∈ `{'RNA','Protein','METHYL'}`, plus `'TEXT'` for `_T`)
   - `checkpoint['fusion_model']` — `OmniFusionBlock` state-dict
   - keys may carry a `module.` prefix from DDP; strip it before loading.

5. **OMIX-T extra hop.** The text-integrated checkpoint additionally
   requires `omix/Youtu_embedding/` (clinical text encoder) and may need
   the PEFT key-remapping helper `_fix_peft_keys` from
   `code/crossmodal_retrieval/pretraining_validation_retrival_stage1_save_embedding.py`
   when LoRA was enabled.

6. **GPU index is hard-coded** in every downstream script (`gpu_index = 1`
   or `gpu_index = 7`). Always tell the user to edit this for their machine
   before launching.

## Common errors quick-reference

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: omix` | missing `PYTHONPATH=../` | prepend it to the python command |
| `ModuleNotFoundError: fmoe` | `fastmoe` not built | `cd fastmoe && USE_NCCL=0 python setup.py install` (needs CUDA); see [official guide](https://github.com/laekov/fastmoe/blob/master/doc/installation-guide.md) |
| `RuntimeError: size mismatch` in `load_state_dict` | wrong `n_input_bins` / `layer_size` | read `args.json` and pass the same values to `PerformerModel(...)` / `OmniFusionBlock(...)` |
| `KeyError: 'RNA'` on checkpoint | loaded wrong checkpoint for selected modalities | `omix_o_pretrain` = RPM only; `omix_t_pretrain` = RPMT |
| Loading hangs on `Youtu_embedding/` | submodule not pulled | only required for `_T` checkpoint; if user only needs omics, switch to `omix_o_pretrain` |

## When the user is exploratory

If the user does not yet know which workflow they want, ask which of the
four they need: **patient embedding**, **disease classification**,
**prognosis**, or **drug response**. Do not start running scripts before
confirming, because each pulls different data (`GDAC/` vs `CCLE2019/`) and
loads a different checkpoint.
