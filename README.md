# SmolVLA Finetune on LIBERO

Standalone finetuning of [SmolVLA](https://huggingface.co/lerobot/smolvla_base) on a subset of
the [LIBERO](https://huggingface.co/datasets/HuggingFaceVLA/libero) dataset.

**No LeRobot training framework used** — plain PyTorch training loop, HuggingFace `datasets` for
data loading, and the original SmolVLA source files downloaded directly from GitHub.

---

## Architecture recap

SmolVLA (~450 M parameters) has two components:

| Component | Frozen? | Description |
|---|---|---|
| SigLIP vision encoder | ✅ yes | Encodes 512×512 images → 64 tokens/frame |
| SmolLM2 (16 layers) | ✅ yes | VLM backbone |
| Action expert (16 layers, 0.75× hidden) | ❌ trained | Cross-attends to VLM, predicts flow-matching vector field |
| `state_proj`, `action_*_proj`, `action_time_mlp` | ❌ trained | Input/output projections |

Training objective: **Conditional Flow Matching** (MSE between predicted and target vector fields).

---

## Project structure

```
smolvla_finetune/
├── lerobot/                    ← minimal stubs (no lerobot install needed)
│   ├── configs/                   PreTrainedConfig, FeatureType, PolicyFeature …
│   ├── optim/                     AdamWConfig, CosineDecayWithWarmupSchedulerConfig
│   ├── policies/
│   │   ├── pretrained.py          PreTrainedPolicy base class w/ from_pretrained()
│   │   ├── rtc/                   RTCProcessor stub (only needed for deployment)
│   │   └── smolvla/           ← SmolVLA source (downloaded by setup.sh)
│   │       ├── modeling_smolvla.py
│   │       ├── configuration_smolvla.py
│   │       └── smolvlm_with_expert.py
│   ├── processor/                 Stub (not used during training)
│   └── utils/                     constants, device_utils, hub
├── dataset.py                  ← PyTorch Dataset for LIBERO
├── normalize.py                ← compute action/state mean-std
├── download_libero.py          ← download a small LIBERO subset from HF Hub
├── train.py                    ← main training loop
├── environment.yml             ← conda environment
└── setup.sh                    ← one-shot setup (downloads SmolVLA source + pip installs)
```

---

## Setup (WSL + conda)

### 1. Create conda environment

```bash
conda env create -f environment.yml
conda activate smolvla
```

### 2. Run setup script (downloads SmolVLA source + pip packages)

```bash
bash setup.sh
```

This downloads `modeling_smolvla.py`, `configuration_smolvla.py`, and `smolvlm_with_expert.py`
from the official [huggingface/lerobot](https://github.com/huggingface/lerobot) repo into
`lerobot/policies/smolvla/`.

### 3. Log in to HuggingFace (needed to download model weights)

```bash
huggingface-cli login
```

---

## Running

### Step 1 — Download LIBERO subset

```bash
# Downloads first 50 episodes of libero_spatial (task_index 0-9)
python download_libero.py --episodes 50 --tasks 0-9 --out data/libero_subset

# Larger subset (all 40 tasks, 200 episodes)
python download_libero.py --episodes 200 --tasks 0-39 --out data/libero_large
```

### Step 2 — Compute normalization statistics

```bash
python normalize.py --data data/libero_subset
# writes data/libero_subset/norm_stats.json
```

### Step 3 — Finetune

```bash
# Recommended: start from pretrained SmolVLA base, train only the action expert
python train.py \
    --data              data/libero_subset \
    --output            checkpoints/run1 \
    --from_pretrained   lerobot/smolvla_base \
    --steps             5000 \
    --batch             4 \
    --lr                1e-4 \
    --chunk             50 \
    --device            cuda

# Train everything (VLM + expert) — needs more GPU memory
python train.py \
    --data              data/libero_subset \
    --output            checkpoints/run_full \
    --from_pretrained   lerobot/smolvla_base \
    --steps             10000 \
    --batch             2 \
    --lr                5e-5 \
    --no-train_expert_only
```

### GPU memory guide

| Config | Approx VRAM |
|---|---|
| `--train_expert_only` (default), batch 4 | ~16 GB |
| `--train_expert_only`, batch 2 | ~10 GB |
| Full model fine-tune, batch 2 | ~24 GB |

---

## Key train.py flags

| Flag | Default | Description |
|---|---|---|
| `--data` | `data/libero_subset` | Path to downloaded LIBERO directory |
| `--from_pretrained` | `lerobot/smolvla_base` | HF repo or local dir for initial weights |
| `--steps` | 5000 | Total gradient steps |
| `--batch` | 4 | Batch size |
| `--lr` | 1e-4 | Peak learning rate |
| `--chunk` | 50 | Action chunk size (50 steps = 5 s at 10 Hz) |
| `--warmup_steps` | 200 | Linear LR warmup |
| `--train_expert_only` | True | Freeze VLM, train only action expert |
| `--save_every` | 500 | Save checkpoint every N steps |
| `--val_every` | 500 | Run validation every N steps |

---

## Notes

- **Images** are passed as float32 `[0, 1]` tensors. SmolVLA's `prepare_images()` internally
  rescales to `[-1, 1]` and pads/resizes to `512×512` for SigLIP.
- **Action chunking**: each training sample contains 50 future actions. End-of-episode
  positions are zero-padded and masked via `action_is_pad`.
- **Language tokens**: task description tokenized to length 48 using the SmolVLM2 tokenizer.
  Descriptions must end with `\n`.
- **Normalization**: action and state are MEAN_STD normalized. Run `normalize.py` first.
