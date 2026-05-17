# Extracting patient embeddings with O-MIX

This file is read by the agent when the user asks for **patient-level
embeddings from a pretrained O-MIX checkpoint**, i.e. inference without
fine-tuning. The reference implementation is
`code/crossmodal_retrieval/pretraining_validation_retrival_stage1_save_embedding.py`.

## Choose the checkpoint

| Goal | Checkpoint | Modalities |
|---|---|---|
| RNA + Protein + Methylation only | `code/pretrain/save/omix_o_pretrain/model_e8.pt` | `RPM` |
| RNA + Protein + Methylation + clinical text | `code/pretrain/save/omix_t_pretrain/model_e5.pt` | `RPMT` |

If the user has no clinical narratives, force them to the `_O` checkpoint —
the `_T` checkpoint additionally requires the `omix/Youtu_embedding/` text
encoder and a tokenized narrative.

## Minimal embedding script

The agent should generate the following template, parameterized to the user's
data path. It builds the same architecture used in
`code/finetune/disease_classification.py` (L760–L797) and reuses the
checkpoint's `args.json` so shapes always match the trained model.

```python
import json, sys
from pathlib import Path
import torch

sys.path.append("../")  # so `omix.*` resolves; run from a sibling of code/omix/
from omix.model.model_multi import PerformerModel
from omix.model.omni_fusion import OmniFusionBlock
from omix.tokenizer import GeneVocab

MODEL_DIR = Path("../pretrain/save/omix_o_pretrain")
CKPT_NAME = "model_e8.pt"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open(MODEL_DIR / "args.json") as f:
    cfg = json.load(f)

vocab_dict = {}
for mod, fname in [("RNA", "vocab_rna.json"),
                   ("Protein", "vocab_protein.json"),
                   ("METHYL", "vocab_methyl.json")]:
    v = GeneVocab.from_file(MODEL_DIR / fname)
    for s in [cfg["pad_token"], "<cls>", "<eoc>"]:
        if s not in v:
            v.append_token(s)
    vocab_dict[mod] = v

encoder_dict = {}
for mod in ["RNA", "Protein", "METHYL"]:
    encoder_dict[mod] = PerformerModel(
        ntoken=len(vocab_dict[mod]),
        d_model=cfg["layer_size"],
        nhead=cfg["nhead"],
        d_hid=cfg["layer_size"],
        nlayers=cfg["nlayers"],
        vocab=vocab_dict[mod],
        pad_token=cfg["pad_token"],
        pad_value=cfg["pad_value"],
        n_input_bins=cfg["n_bins"] + 2,
        cell_emb_style="cls",
        input_emb_style="category",
        dropout=cfg["dropout"],
    ).to(DEVICE).eval()

vocab_mod = cfg["modality_dict"].copy()
if "<pad>" not in vocab_mod:
    vocab_mod["<pad>"] = len(vocab_mod)

fusion = OmniFusionBlock(
    cfg["num_modalities"], 0, 3,
    cfg["layer_size"], cfg["num_layers_fus"],
    cfg["num_experts"], cfg["num_routers"],
    cfg["top_k"], cfg["nhead"],
    cfg["dropout"], vocab_mod=vocab_mod, device=DEVICE,
).to(DEVICE).eval()

ckpt = torch.load(MODEL_DIR / CKPT_NAME, map_location="cpu")

def _strip_module(sd):
    return {k.replace("module.", ""): v for k, v in sd.items()}

for mod in ["RNA", "Protein", "METHYL"]:
    encoder_dict[mod].load_state_dict(_strip_module(ckpt["encoder_dict"][mod]),
                                      strict=True)
fusion.load_state_dict(_strip_module(ckpt["fusion_model"]), strict=True)
```

## Feeding real data through

The encoders consume **tokenized + binned** omics data, not raw csv values.
Reuse the helpers already in the repo:

- For RNA / Protein matrices held in an `AnnData`:
  ```python
  from omix.preprocess_bulk import Preprocessor as PreprocessorRNA
  from omix.preprocess_rppa import Preprocessor as PreprocessorProtein
  from omix.tokenizer import tokenize_and_pad_batch
  ```
- For methylation, the scripts wrap a `pandas.DataFrame` into
  `anndata.AnnData(df)` and tokenize directly (see `tokenize_omics(...)` in
  any of the `code/finetune/*.py` files — search for `def tokenize_omics`).

Maximum sequence lengths the agent must pass to `tokenize_and_pad_batch`:

```python
MAX_SEQ_LEN_RNA     = cfg["max_seq_len_rna"]      # 19206
MAX_SEQ_LEN_PROTEIN = cfg["max_seq_len_protein"]  # 17008
MAX_SEQ_LEN_METHYL  = cfg["max_seq_len_methyl"]   # 12921
```

## Reading the fused embedding

After encoding each modality the agent feeds the per-modality CLS embeddings
into the fusion block. The exact call signature lives in
`code/crossmodal_retrieval/pretraining_validation_retrival_stage1_save_embedding.py`
(`process_modality` and the surrounding loop). The relevant pattern is:

1. Per modality `mod`: `z_mod = encoder_dict[mod](src, values, ...)`, take the
   CLS slice → shape `[B, layer_size]`.
2. Stack the per-modality CLS tensors plus the learnable "missing-modality"
   token (loaded via `ClassificationPredictor.load_missing_embeds(ckpt, ...)`
   in the finetune scripts) into a `[B, num_modalities, layer_size]` tensor.
3. Call `fusion(...)` → returns a fused patient embedding `[B, layer_size]`,
   which is the recommended representation for downstream tasks (sklearn
   classifier, UMAP, retrieval).

If the user only has **one or two of the three omics**, the missing slots
should be filled with the learnable token pulled from the checkpoint:

```python
missing = ckpt.get("missing_embeds", None)   # dict[mod -> tensor[1, d_model]]
```

(present in `omix_o_pretrain/model_e8.pt`).

## Sanity checklist before claiming success

- `torch.load` finishes without `KeyError`. If it complains about a missing
  key, the agent loaded the wrong checkpoint for the chosen modality string.
- All `load_state_dict(..., strict=True)` calls return empty
  missing/unexpected key lists. Any mismatch usually means `layer_size`,
  `nlayers`, or `n_input_bins` was set differently from `args.json`.
- The fused embedding has shape `[batch, cfg["layer_size"]] = [batch, 512]`.
- For a quick smoke test on a single sample, the agent can replace real data
  with a dummy tensor of the right shape — but should warn the user this
  only validates the pipeline, not the biological signal.

## When **not** to use this template

- Stage-I single-modality checkpoints (`rna_pretrain`, `protein_pretrain`,
  `methylation_pretrain`) — those are built with `model_uni.PerformerModel`
  and have no `OmniFusionBlock`. For those, instantiate only the
  corresponding encoder and read `cls_emb` directly.
- The `_T` checkpoint with frozen Youtu text encoder — additionally
  instantiate `TrainableTextEncoder` from `omix.model.omni_fusion` (see
  the retrieval script) and ensure `omix/Youtu_embedding/` is reachable.
