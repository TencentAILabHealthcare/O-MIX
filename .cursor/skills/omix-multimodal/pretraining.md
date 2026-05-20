# Stage I & II pretraining

This file describes how to **retrain O-MIX from scratch**. For routine
inference / fine-tuning the user should reuse the existing checkpoints
(`omix_o_pretrain/model_e8.pt` or `omix_t_pretrain/model_e5.pt`) — those
are documented in `embedding.md` and `finetune.md`.

Pretraining is expensive (multi-GPU, days of wallclock). Confirm with the
user before triggering it.

## Pretraining data (not open-sourced)

The manuscript describes the Stage I/II corpus; it is **not** bundled or
linked for download in `README.md`. Several scripts still default to paths
under `data/pretraining_data/` (e.g. `rna_full_data.h5ad`,
`generated_files/textual_annotations_*.json`). The user must:

1. Obtain or rebuild equivalent inputs per the paper, **or**
2. Edit `input_h5ad_paths` / `DATA_PATHS` / `TEXT_JSON_PATH` in the relevant
   script to point at their own files.

For LLM clinical narratives on a **new** cohort, use `README.md` §§1–2
(`code/data_preprocess/` + `vllm` env). Do not instruct users to download
a `pretraining_data` archive from this repository.

## Stage layout

| Stage | Script | Modality | Inputs | Outputs |
|---|---|---|---|---|
| I | `pretrain/pretrain_RNA.py` | RNA only | bulk RNA AnnData | `pretrain/save/rna_pretrain/model_e*.pt` |
| I | `pretrain/pretrain_protein.py` | Protein only | RPPA AnnData | `pretrain/save/protein_pretrain/model_e*.pt` |
| I | `pretrain/pretrain_Methylation.py` | Methylation only | methylation AnnData | `pretrain/save/methylation_pretrain/model_e*.pt` |
| II | `pretrain/pretrain_OMIX_O.py` | RNA + Protein + Methyl | paired multi-omics + Stage-I weights | `pretrain/save/omix_o_pretrain/model_e*.pt` |
| II | `pretrain/pretrain_OMIX_T.py` | RNA + Protein + Methyl + clinical text | paired + clinical narratives + Stage-I weights | `pretrain/save/omix_t_pretrain/model_e*.pt` |

Stage II loads the Stage-I checkpoints via `load_model_RNA`, `load_model_Protein`,
`load_model_methyl` — these paths are at the top of `pretrain_OMIX_O.py`
and `pretrain_OMIX_T.py` and default to the three Stage-I directories.

### Stage-I epoch filenames are **hardcoded** in Stage-II scripts

This is the #1 silent-failure trap. `pretrain_OMIX_O.py` and
`pretrain_OMIX_T.py` look for these **exact** filenames inside each
`load_model_*` directory:

| Modality | Hardcoded filename | Source line |
|---|---|---|
| RNA | `model_e15.pt` | `pretrain_OMIX_O.py` L207 |
| Protein | `model_e5.pt` | L240 |
| Methylation | `model_e40.pt` | L270 |

If the user retrains Stage I and ends up with e.g. `model_e20.pt`, Stage II
will not find it. Workarounds (in order of preference):

1. Rename / symlink the chosen epoch:
   `ln -s model_e20.pt save/rna_pretrain/model_e15.pt`
2. Or edit the hardcoded `model_e<N>.pt` strings in `pretrain_OMIX_O.py` to
   match the epoch the user actually wants to use.

The corresponding `vocab.json` and `args.json` in each Stage-I save folder
are picked up by directory, so they don't have this issue.

## Launch commands

All five scripts use **DDP via `torchrun`**, except `pretrain_protein.py`
which runs on a single GPU because the protein dataset is small.

```bash
conda activate omix
cd code/pretrain

PYTHONPATH=../ torchrun --nnodes=1 --nproc_per_node=4 pretrain_RNA.py

PYTHONPATH=../ CUDA_VISIBLE_DEVICES=0 python pretrain_protein.py

PYTHONPATH=../ torchrun --nnodes=1 --nproc_per_node=4 pretrain_Methylation.py

PYTHONPATH=../ torchrun --nnodes=1 --nproc_per_node=4 pretrain_OMIX_O.py

PYTHONPATH=../ torchrun --nnodes=1 --nproc_per_node=4 pretrain_OMIX_T.py
```

Change `--nproc_per_node` to match the available GPU count on the user's
node. The dataloader sampler (`DistributedSubsetsBatchSampler`) adapts
automatically.

## Key hyperparameters (Stage II `pretrain_OMIX_O.py`)

All defaults live in the `hyperparameter_defaults` dict at L57–L117. The
important knobs:

| Knob | Default | Purpose |
|---|---|---|
| `epochs` | `10` | Stage-II runs short because encoders are frozen. |
| `lr` | `1e-4` | AdamW base LR. |
| `batch_size` | `32` | Per-GPU. Effective batch = 32 × `nproc` × `gradient_accumulation_steps`. |
| `gradient_accumulation_steps` | `2` | Raise to fit larger effective batch on memory-constrained nodes. |
| `layer_size` | `512` | Encoder + fusion d_model. **Changing this means Stage-I checkpoints no longer load.** |
| `nlayers` / `nhead` | `6` / `8` | Transformer depth / heads. Same constraint as `layer_size`. |
| `n_bins` | `51` | Expression value bins. Pinned to the pretrained vocab. |
| `num_experts` / `num_routers` / `top_k` | `16 / 1 / 4` | MoE config in OmniFusionBlock. |
| `simsiam_loss_weight` | `10` | Cross-modal contrastive alignment weight. |
| `protein_loss_weight` | `5` | Up-weights the smaller protein modality. |
| `frozen_encoder` | `True` | If `True`, Stage-I encoders are frozen and only the fusion + projection heads train. Flip to `False` for full Stage-II fine-tune (more expensive). |
| `mask_ratio` | `0.15` | MLM masking proportion for GEP / GEPC objectives. |

## Pretraining objectives (active by default in Stage II)

- **GEP** — masked gene expression prediction
- **GEPC** — gene expression prediction conditional on cell embedding
- **CLS / ESC / DAR** — off by default; flip to `True` for ablations
- **SimCLR / MoCo-style** — cross-modal contrastive (controlled by
  `simsiam_loss_weight`)

Loss weights are summed; tune one weight at a time to avoid drift.

## OMIX-T extras (`pretrain_OMIX_T.py`)

- Adds `TEXT` to `modality_dict` and bumps `num_modalities` to `4`.
- Requires per-sample `textual_annotations_*.json` (default path in script:
  `data/pretraining_data/generated_files/` — user-supplied; generate via
  `README.md` §2 or provide equivalents). Narrative generation itself is
  out of scope for this Skill file; see `README.md` §§1–2.
- Loads `omix/Youtu_embedding/` as the trainable text encoder.
- Often runs with LoRA on the text branch — see `use_LoRA` /
  `text_only_lora` flags in the saved `args.json` of an existing checkpoint.

## Stage I quickstart (RNA example)

```bash
cd code/pretrain
PYTHONPATH=../ torchrun --nnodes=1 --nproc_per_node=4 pretrain_RNA.py
```

Edit the top of `pretrain_RNA.py` for:
- `dataset_name` — used in the save folder name.
- `input_h5ad_paths` — list of paired AnnData files for pretraining.
- `epochs`, `lr`, `batch_size` — same conventions as Stage II.

Stage I has **no fusion block**, so `OmniFusionBlock` is not constructed
and `fastmoe` is not required for Stage I scripts. (`fastmoe` only enters
at Stage II.)

## Monitoring and resuming

- **wandb** is wired up by default (`wandb.init(...)`). Set
  `WANDB_MODE=offline` to keep logs locally without internet.
- **Save folder name is timestamped**, not the clean `<dataset_name>`.
  `pretrain_OMIX_O.py` L162 builds the save dir as
  `save/dev_<dataset_name>-DDP-<MonDD-HH-MM>/`, e.g.
  `save/dev_omix_o_pretrain-DDP-Apr07-21-30/`. **Downstream scripts in
  this Skill (embedding / finetune / clustering / retrieval) assume the
  clean name `save/omix_o_pretrain/`** — after pretraining finishes, the
  user should rename / symlink the timestamped folder to the clean name:
  ```bash
  cd code/pretrain/save
  ln -s dev_omix_o_pretrain-DDP-Apr07-21-30 omix_o_pretrain
  ```
- Checkpoints inside the folder are `model_e<N>.pt`, one per epoch. To
  resume from a specific epoch, set `load_model` in the script to that path.
- `pretraining_dataset_split.json` is written once per run and pins the
  train / val IDs — keep this file with the checkpoint, the retrieval and
  validation scripts depend on it.

## Gotchas

1. **`fastmoe` build must match CUDA / PyTorch.** Stage II import will
   crash with `ModuleNotFoundError: fmoe` or a CUDA mismatch at first
   forward pass. Rebuild with `cd fastmoe && pip install -e .` against the
   same CUDA version as installed `torch`.
2. **DDP rank-0-only side effects.** Writes to disk (wandb logs, checkpoint
   dumps) must be guarded by `if dist.get_rank() == 0:`. The shipped
   scripts already do this — don't add new write calls without the guard.
3. **Vocab files are pinned at first save.** Once `vocab_<mod>.json` is
   written into the save folder, all downstream scripts assume those exact
   token sets. Re-running with a different gene panel requires a fresh
   save folder.
4. **`DistributedSubsetsBatchSampler` partitions by dataset**, not by
   sample. If one of the input AnnData files is much smaller than the
   others it will be over-sampled — check the `dataset_sizes` printout at
   startup.
5. **Stage II encoder freeze.** `frozen_encoder=True` (default) means the
   Stage-I weights stay fixed. If the user reports "no improvement after
   epoch 1", check that this flag is `False` if they intended to fine-tune
   the encoders together.

## When the user just wants to ablate one modality

Rather than rerunning Stage II from scratch, suggest:
- Edit the OPTION string in the relevant downstream script (`embedding.md`
  / `finetune.md`) to drop the modality.
- The fusion block already supports missing modalities via the learned
  `missing_embeds` tokens — no retraining required.
