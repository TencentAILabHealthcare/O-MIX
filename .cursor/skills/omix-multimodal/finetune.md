# Fine-tuning O-MIX downstream tasks

Three downstream tasks are supported. Each ships as a single self-contained
script in `code/finetune/`. All three follow the same general loop:

1. Build per-modality `PerformerModel` + `OmniFusionBlock` from `args.json`.
2. Load Stage-II checkpoint into encoders + fusion.
3. Attach a task-specific head, train with k-fold CV, log mean ± std.

## Common pre-flight check the agent must run

Before launching **any** of the three scripts:

- [ ] `conda activate omix`
- [ ] `cd OMIX_codebuddy/code/finetune`
- [ ] Edit `gpu_index` near the top of the chosen script (default is 1 or 7
      depending on script). Set to a GPU that exists and is free.
- [ ] Verify the data root is populated:
      - For disease / prognosis: `ls ../../data/GDAC/BRCA/` should show
        per-modality `h5ad` or csv files.
      - For drug response: `ls ../../data/CCLE2019/preprocessed/` should list
        the drug subfolders and `textual_annotations_ccle2019.json`.
- [ ] Confirm the checkpoint:
      - `omix_o_pretrain/model_e8.pt` for disease & prognosis (RPM)
      - `omix_t_pretrain/model_e5.pt` for drug response (RPMT)
- [ ] Launch with `PYTHONPATH=../`.

If anything is missing, the script will fail late inside the dataloader.
Catching it pre-flight saves the user 5–10 minutes per attempted run.

---

## §1 Pan-cancer disease classification

Script: `code/finetune/disease_classification.py`

| Knob | Default | What to change for |
|---|---|---|
| `OPTION` | `'RPM'` | Drop letters to ablate modalities (e.g. `'RP'` for RNA+Protein) |
| `DISEASE_NAME_LIST` | 30 TCGA types | Restrict to a subset for fast smoke tests, e.g. `['BRCA','LUAD','LGG']` |
| `DATA_ROOT_DIR` | `'../../data/GDAC'` | Point at a different cohort directory |
| `LOAD_MODEL_PATH` | `'../pretrain/save/omix_o_pretrain'` | Switch to a different Stage-II run |
| `model_epoch` | `'model_e8.pt'` | Earlier epoch for comparison |
| `BATCH_SIZE` / `GRADIENT_ACCUMULATION_STEPS` | `4 / 4` | Reduce on small GPUs |
| `EPOCHS` | `100` | Lower for debugging |
| `freeze_omics` / `freeze_fusion` | `False / False` | `True / True` to use O-MIX as a frozen feature extractor + linear head only |
| `N_SPLITS` | `5` | k-fold count |

Launch:

```bash
cd code/finetune
PYTHONPATH=../ python -u disease_classification.py 2>&1 | tee disease_run.log
```

Outputs are written to:

```
classification_file_dir_novel/classification-LEARNING_RATE{lr}-DROPOUT{do}-BATCH_SIZE{bs}-GRADIENT_ACCUMULATION_STEPS{gas}_hvg{use_hvg}_gateloss{use_gate_loss}/
```

Final report prints mean ± std across folds for **ACC, F1, Recall, Precision,
AUPR, AUC** plus an aggregated `classification_report` from sklearn.

---

## §2 Prognosis / survival prediction

Script: `code/finetune/prognosis_prediction.py`

| Knob | Default | Notes |
|---|---|---|
| `DISEASE_NAME_LIST` | `['OV']` | The script trains **one cancer at a time**. Common choices: `BRCA`, `COAD`, `LGG`, `OV`. |
| `OPTION` | `'RPM'` | **Display string only — does not control modalities.** See note below. |
| `LOAD_MODEL_PATH` | `'../pretrain/save/omix_o_pretrain'` | RPM checkpoint required. |
| `BATCH_SIZE` | `4` | Cox loss benefits from larger batches; raise if VRAM allows. |
| `EPOCHS` | `50` | |
| `N_SPLITS` | `5` | Stratified by survival event indicator. |

### `OPTION` is cosmetic — actual modality control is in `MODALITIES_ORDER`

This trap differs from §1 disease classification. In
`prognosis_prediction.py` L717, the call site is hardcoded to:

```python
omics_data, metadata_clean, final_sample_ids = create_flexmoe_input_for_cv(
    sample_ids_union, metadata_df, raw_mod_data,
    MODALITIES_ORDER, MODALITIES_ORDER,   # ← modalities_to_keep == MODALITIES_ORDER
    ...
)
```

So changing `OPTION = 'RP'` does **not** drop methylation — it only changes a
`print(...)` statement at the top. To truly ablate a modality, edit
`MODALITIES_ORDER` itself (e.g. `['RNA', 'Protein']`), which propagates to
both the model build loop and the dataset construction.

Launch:

```bash
cd code/finetune
PYTHONPATH=../ python -u prognosis_prediction.py 2>&1 | tee prognosis_run.log
```

Outputs go to `survival_file_dir_novel/survival_multimodal_finetune-performer-{disease}-gate{...}-hvg{...}-DROPOUT{...}/`.

Primary reported metric is **concordance index (C-index)** from
`lifelines.utils.concordance_index`, averaged over folds. Higher = better.

To benchmark several cancers, the agent should instruct the user to edit
`DISEASE_NAME_LIST` to a single cancer and **launch the script once per
cancer** rather than batching them — the script does not aggregate across
cohorts.

---

## §3 Drug response prediction (CCLE)

Script: `code/finetune/drug_response_prediction.py`

This is the **only** downstream task that uses the `_T` checkpoint
(`omix_t_pretrain/model_e5.pt`) and therefore the only one that touches the
clinical text encoder.

| Knob | Default | Notes |
|---|---|---|
| `OPTION` | `'RPMT'` | Must include all four modalities for the `_T` checkpoint. |
| `MODALITIES_TO_KEEP` | `['RNA', 'Protein', 'METHYL', 'TEXT']` | Subsets supported: `RPM`, `RPT`, `RMT`, `PMT` (see comment in script L52). |
| `DRUG_NAME` | `'ERLOTINIB'` | Available: `LAPATINIB`, `ERLOTINIB`, `PACLITAXEL`, `NILOTINIB`, `SORAFENIB`, `IRINOTECAN`, `TOPOTECAN`. |
| `LOAD_MODEL_PATH` | `'../pretrain/save/omix_t_pretrain'` | RPMT checkpoint required. |
| `TEXT_EMBEDDING_FILE` | `'../../data/pretraining_data/generated_files/textual_annotations_ccle2019.json'` | Default path only — **not** shipped with the repo. User must place LLM-generated CCLE narratives here (see `README.md` §2) or retarget the knob to their JSON. |
| `freeze_omics` / `freeze_text` | `True / True` | Default fine-tunes only the fusion + head. Flip to `False` for full unfreeze. |
| `BATCH_SIZE` / `GRADIENT_ACCUMULATION_STEPS` | `1 / 16` | Larger effective batch via accumulation; text encoder is memory-heavy. |
| `EPOCHS` | `100` | |

Launch:

```bash
cd code/finetune
PYTHONPATH=../ python -u drug_response_prediction.py 2>&1 | tee drug_run.log
```

Primary metrics: **Pearson r, Spearman ρ, MSE, R²** between predicted and
ground-truth IC50 / AUC values, reported as mean ± std over folds.

LoRA: the script imports `peft.LoraConfig`. Whether LoRA is applied depends
on `use_LoRA` inside the saved `args.json` of the checkpoint. If LoRA was
on at pretraining, the user does **not** need to add new LoRA configs; if
loading fails with PEFT key mismatches, copy `_fix_peft_keys` from
`code/crossmodal_retrieval/pretraining_validation_retrival_stage1_save_embedding.py`
into the script.

---

## Fast smoke-test recipe (for any of the three)

When the user just wants to verify their setup works before committing to a
full 100-epoch run:

1. Shrink the cohort: edit `DISEASE_NAME_LIST` to a single cancer (or pick a
   drug with few cell lines like `TOPOTECAN`).
2. `EPOCHS = 1`, `N_SPLITS = 2`.
3. Optional: cap dataset size by slicing `Subset(full_dataset, indices[:32])`
   in the fold loop.
4. Confirm the run reaches "Final test metrics" within a few minutes; that
   proves the data path, checkpoint, and modality string are all consistent.

Once smoke test passes, revert the three values and launch the real run with
`tee`'d logs.

---

## Where results live

| Task | Output dir prefix |
|---|---|
| Disease classification | `code/finetune/classification_file_dir_novel/` |
| Prognosis | `code/finetune/survival_file_dir_novel/` |
| Drug response | `code/finetune/drug_response_file_dir_novel/` (subdir keyed by `DRUG_NAME`) |

Each run writes per-fold checkpoints and a summary file. The agent should
remind the user to clear or rename these dirs before re-running with the
same hyperparameters — otherwise mixed-history results may end up in the
same folder.
