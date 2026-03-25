# SmolVLA Finetuning on LIBERO — Complete Guide

---

## Table of Contents

1. [What is SmolVLA?](#1-what-is-smolvla)
2. [Project Overview](#2-project-overview)
3. [Architecture Deep Dive](#3-architecture-deep-dive)
4. [Training Objective — Flow Matching](#4-training-objective--flow-matching)
5. [Project Structure](#5-project-structure)
6. [How the Stubs Work](#6-how-the-stubs-work)
7. [The LIBERO Dataset](#7-the-libero-dataset)
8. [Setup from Scratch](#8-setup-from-scratch)
9. [Running Everything Step by Step](#9-running-everything-step-by-step)
10. [Understanding Each Script](#10-understanding-each-script)
11. [Training Arguments Reference](#11-training-arguments-reference)
12. [GPU Memory Guide](#12-gpu-memory-guide)
13. [What Gets Trained vs Frozen](#13-what-gets-trained-vs-frozen)
14. [Batch Format SmolVLA Expects](#14-batch-format-smolvla-expects)
15. [Bugs Fixed During Debugging](#15-bugs-fixed-during-debugging)
16. [Checkpoints and Resuming](#16-checkpoints-and-resuming)

---

## 1. What is SmolVLA?

SmolVLA (**Small Vision-Language-Action** model) is a ~450M parameter robot policy model
released by HuggingFace in 2025 ([paper](https://arxiv.org/abs/2506.01844)).

It takes as input:
- **Camera images** (one or more RGB images from the robot)
- **Robot state** (joint positions / end-effector pose)
- **Language instruction** ("pick up the red cube and place it in the basket")

And outputs:
- **A chunk of future actions** (e.g., the next 50 joint position commands)

It is built on top of **SmolVLM2-500M-Video-Instruct** (a compact Vision-Language Model)
with an additional **Action Expert** transformer that predicts robot actions using
**Flow Matching** (a generative modelling technique).

The pretrained weights are at: `lerobot/smolvla_base` on HuggingFace Hub.

---

## 2. Project Overview

**Goal:** Finetune SmolVLA on the LIBERO robot manipulation dataset, without using the
LeRobot training framework. Everything is plain PyTorch.

**Why not use lerobot-train?**
The official `lerobot-train` command has many opinions about dataset format, environment
wrappers, evaluation loops, etc. This project gives you full control over:
- How data is loaded and preprocessed
- The training loop (optimizer, scheduler, logging)
- What parts of the model are trained

**What this project does NOT install:** The full `lerobot` Python package. Instead, it
downloads only the 3 SmolVLA model source files directly from GitHub and provides
minimal stub modules for the lerobot internals they depend on.

---

## 3. Architecture Deep Dive

SmolVLA has two main components that run jointly:

```
Input: Images + State + Language
            │
     ┌──────▼──────────────────────────────────────┐
     │           VLM Backbone (FROZEN)              │
     │                                              │
     │  ┌─────────────┐   ┌──────────────────────┐ │
     │  │ SigLIP      │   │  SmolLM2 (16 layers) │ │
     │  │ Vision Enc. │──▶│  Language Model      │ │
     │  │ (frozen)    │   │  (frozen)            │ │
     │  └─────────────┘   └──────────┬───────────┘ │
     └─────────────────────────────  │  ────────────┘
                                     │ KV cache / keys+values
     ┌───────────────────────────────▼─────────────┐
     │         Action Expert (TRAINED)              │
     │                                              │
     │  Noisy actions + timestep embedding          │
     │         │                                    │
     │  ┌──────▼──────────────────────────────────┐ │
     │  │  16 layers (0.75× hidden size of VLM)   │ │
     │  │  Even layers: cross-attention → VLM KVs │ │
     │  │  Odd  layers: causal self-attention      │ │
     │  └──────┬──────────────────────────────────┘ │
     └─────────│────────────────────────────────────┘
               │
     ┌─────────▼─────┐
     │ action_out_proj│  → predicted vector field v_t
     └───────────────┘
```

### Component sizes

| Component | Parameters | Trainable (default) |
|---|---|---|
| SigLIP vision encoder | ~93M | No (frozen) |
| SmolLM2 text model (16/32 layers) | ~250M | No (frozen) |
| Action Expert (16 layers, 0.75× hidden) | ~99M | **Yes** |
| `state_proj` (Linear 32→vlm_hidden) | tiny | **Yes** |
| `action_in_proj` (Linear 32→expert_hidden) | tiny | **Yes** |
| `action_out_proj` (Linear expert_hidden→32) | tiny | **Yes** |
| `action_time_mlp_in/out` (2-layer MLP) | tiny | **Yes** |

**Total: ~450M parameters, ~99.9M trainable** (when `train_expert_only=True`)

### Image processing

- Images are passed as float32 tensors in `[0, 1]`
- SmolVLA internally rescales to `[-1, 1]` and pads/resizes to `512×512` for SigLIP
- SigLIP compresses each image to **64 visual tokens** (via PixelShuffle)

### Language processing

- Task descriptions are tokenized with the SmolVLM2 tokenizer (max length = 48)
- Descriptions **must end with `\n`** (required by SmolVLA's formatting)
- Language tokens are passed as `int64`, attention mask as `int64`

---

## 4. Training Objective — Flow Matching

SmolVLA uses **Conditional Flow Matching** to learn action distributions.

Instead of predicting actions directly, it predicts a **vector field** that "flows"
random noise toward real actions over multiple denoising steps.

**During training (one step):**

```python
# 1. Sample a random timestep t ~ Beta(1.5, 1.0) in (0, 1)
time = sample from Beta distribution

# 2. Interpolate between real actions and random noise
#    x_t = t * noise + (1 - t) * real_actions
x_t = time * noise + (1 - time) * actions

# 3. The target vector field (what we want the model to predict)
u_t = noise - actions

# 4. Run forward pass: model predicts v_t (the vector field at x_t, time t)
v_t = model(images, state, language, x_t, time)

# 5. Loss = MSE between predicted and target vector field
loss = MSE(v_t, u_t)   # averaged over non-padded timesteps
```

**During inference (denoising loop):**

```python
x_t = random_noise   # start from Gaussian noise
dt = -1.0 / 10       # 10 denoising steps

for step in range(10):
    time = 1.0 + step * dt     # goes from 1.0 down to ~0.0
    v_t  = model(x_t, time)    # predict vector field (VLM prefix cached)
    x_t  = x_t + dt * v_t     # Euler step

# x_t is now approximately the predicted actions
```

The VLM prefix (images + language + state) is computed **once** and cached as KV pairs,
making inference efficient.

---

## 5. Project Structure

```
smolvla_finetune/
│
├── lerobot/                         ← Minimal stubs (NO lerobot install needed)
│   ├── __init__.py
│   ├── configs/
│   │   ├── policies.py              ← PreTrainedConfig base class
│   │   │                               • dataclass with type, device, tags…
│   │   │                               • register_subclass() decorator
│   │   │                               • from_pretrained() loads from HF Hub
│   │   │                               • image_features / action_feature properties
│   │   ├── types.py                 ← FeatureType, NormalizationMode, PolicyFeature
│   │   └── train.py                 ← TrainPipelineConfig (empty stub)
│   │
│   ├── optim/
│   │   ├── optimizers.py            ← AdamWConfig (stub dataclass)
│   │   └── schedulers.py            ← CosineDecayWithWarmupSchedulerConfig (stub)
│   │
│   ├── policies/
│   │   ├── pretrained.py            ← PreTrainedPolicy base class
│   │   │                               • extends nn.Module
│   │   │                               • from_pretrained() downloads safetensors
│   │   │                               • save_pretrained() saves weights + config
│   │   ├── utils.py                 ← populate_queues, log_model_loading_keys
│   │   ├── rtc/
│   │   │   ├── configuration_rtc.py ← RTCConfig (stub, only for deployment)
│   │   │   └── modeling_rtc.py      ← RTCProcessor (stub, only for deployment)
│   │   └── smolvla/                 ← REAL SmolVLA source (downloaded from GitHub)
│   │       ├── __init__.py
│   │       ├── modeling_smolvla.py      ← SmolVLAPolicy + VLAFlowMatching
│   │       ├── configuration_smolvla.py ← SmolVLAConfig (all hyperparameters)
│   │       └── smolvlm_with_expert.py   ← SmolVLMWithExpertModel (joint VLM+expert)
│   │
│   ├── processor/
│   │   ├── __init__.py              ← Empty stubs (processor not used in training)
│   │   └── converters.py            ← Empty stubs
│   │
│   └── utils/
│       ├── constants.py             ← All key name constants (OBS_STATE, ACTION…)
│       ├── device_utils.py          ← get_safe_dtype()
│       └── hub.py                   ← HubMixin (stub)
│
├── download_libero.py               ← Stream LIBERO from HF Hub, save as Parquet
├── normalize.py                     ← Compute mean/std for actions and states
├── dataset.py                       ← PyTorch Dataset (Parquet → SmolVLA batch)
├── train.py                         ← Complete training loop
├── environment.yml                  ← Conda environment definition
├── setup.sh                         ← One-shot setup script
├── README.md                        ← Quick start
└── GUIDE.md                         ← This file
```

---

## 6. How the Stubs Work

SmolVLA's source code (`modeling_smolvla.py`, etc.) was written as part of the `lerobot`
Python package. It imports things like:

```python
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.configs.types import FeatureType, PolicyFeature
```

Rather than installing all of lerobot (which pulls in robot simulators, gym environments,
etc.), this project places a `lerobot/` folder **in the same directory** as `train.py`.
Python's import system finds this local folder first, so all imports resolve to our
lightweight stub implementations.

**What the stubs provide:**
- `PreTrainedConfig` — a Python dataclass with `from_pretrained()` that downloads
  `config.json` from HuggingFace Hub and reconstructs the config object
- `PreTrainedPolicy` — a plain `nn.Module` subclass with `from_pretrained()` that
  downloads `model.safetensors` and loads the weights
- `FeatureType`, `PolicyFeature` — simple enums/dataclasses describing input/output shapes
- All string constants (`OBS_STATE = "observation.state"`, etc.)
- `RTCProcessor`, `RTCConfig` — empty stubs (only needed for real-time robot deployment)

**What the stubs do NOT provide:**
- Any training framework (no datasets, no environments, no eval loops)
- The processor pipeline (`processor_smolvla.py`) — we build batches ourselves in `dataset.py`

---

## 7. The LIBERO Dataset

LIBERO is a robot manipulation benchmark with 40 tasks and ~1,693 episodes.
It is available on HuggingFace Hub as `HuggingFaceVLA/libero`.

### Task groups

| Task indices | Group name | Description |
|---|---|---|
| 0–9 | libero_spatial | Pick and place objects into a basket |
| 10–19 | libero_object | Put objects onto a plate |
| 20–29 | libero_goal | Stack objects on top of each other |
| 30–39 | libero_100 | Put objects in a drawer and close it |

### Data format (per frame)

| Field | Type | Shape | Description |
|---|---|---|---|
| `observation.images.image` | PNG bytes | 256×256 RGB | Main camera |
| `observation.images.image2` | PNG bytes | 256×256 RGB | Wrist camera |
| `observation.state` | float32 array | (8,) | Joint positions + gripper |
| `action` | float32 array | (7,) | Target joint positions + gripper |
| `episode_index` | int | — | Which episode (0–1692) |
| `frame_index` | int | — | Frame within episode |
| `task_index` | int | — | Which task (0–39) |

- **State (8-dim):** 6 joint angles + 1 end-effector state + 1 gripper
- **Action (7-dim):** 6 joint targets + 1 gripper command
- **FPS:** 10 Hz (so chunk_size=50 = 5 seconds of future actions)

### Downloading a subset

`download_libero.py` streams the dataset from HuggingFace and stops early once it has
collected enough episodes. Images are converted from PIL to PNG bytes and saved as a
Parquet file for fast local loading.

```bash
# 20 episodes, tasks 0-9 (libero_spatial) — ~100MB
python download_libero.py --episodes 20 --tasks 0-9

# 50 episodes, all tasks — ~250MB
python download_libero.py --episodes 50 --tasks 0-39
```

---

## 8. Setup from Scratch

### Prerequisites
- WSL (Ubuntu) with conda (Miniforge/Miniconda/Anaconda)
- NVIDIA GPU with CUDA 12.x (for GPU training; CPU works but is slow)
- HuggingFace account (free) to download model weights

### Step 1 — Create the conda environment

```bash
cd '/mnt/c/Users/Moham/OneDrive/Desktop/Claude Projects/smolvla_finetune'
conda env create -f environment.yml
conda activate smolvla
```

### Step 2 — Download SmolVLA source files + install pip packages

```bash
bash setup.sh
```

This script:
1. Downloads `modeling_smolvla.py`, `configuration_smolvla.py`, `smolvlm_with_expert.py`
   from the official `huggingface/lerobot` GitHub repo into `lerobot/policies/smolvla/`
2. Installs all pip dependencies (torch, transformers, datasets, safetensors, etc.)

### Step 3 — Log in to HuggingFace

```bash
huggingface-cli login
```

You need this to download the pretrained SmolVLA weights (`lerobot/smolvla_base`).
The weights (~900MB) are cached locally after first download.

---

## 9. Running Everything Step by Step

```bash
# Activate the environment (do this every time you open a new terminal)
conda activate smolvla
cd '/mnt/c/Users/Moham/OneDrive/Desktop/Claude Projects/smolvla_finetune'

# ── Step 1: Download LIBERO data ──────────────────────────────────────────
# Downloads 20 episodes from libero_spatial (tasks 0-9), saves to data/libero_subset/
python download_libero.py --episodes 20 --tasks 0-9 --out data/libero_subset

# ── Step 2: Compute normalization statistics ──────────────────────────────
# Computes mean/std of actions and states from the downloaded data
# Writes: data/libero_subset/norm_stats.json
python normalize.py --data data/libero_subset

# ── Step 3: Train ────────────────────────────────────────────────────────
# Recommended: start from pretrained weights, train only the action expert
python train.py \
    --data            data/libero_subset \
    --output          checkpoints/run1 \
    --from_pretrained lerobot/smolvla_base \
    --steps           5000 \
    --batch           4 \
    --lr              1e-4 \
    --chunk           50 \
    --device          cuda
```

### What you will see during training

```
17:55:31 | INFO | Parameters: 450.0M total, 99.9M trainable
17:55:31 | INFO | Starting training for 5000 steps on cuda
17:55:45 | INFO | step     50/5000 | loss 2.4521 | lr 2.50e-05 | 3.1 steps/s
17:56:02 | INFO | step    100/5000 | loss 2.1034 | lr 5.00e-05 | 3.2 steps/s
...
17:58:21 | INFO |   [val] step 500 | val_loss 1.8832
17:58:21 | INFO |   [ckpt] saved to checkpoints/run1/step_0000500
```

The **loss should decrease** over training. A good sign is loss going below 1.0 after
a few thousand steps on a reasonable dataset size.

---

## 10. Understanding Each Script

### `download_libero.py`

Streams the LIBERO HuggingFace dataset and collects the first N episodes from the
requested task range. Stops early once enough episodes are collected (no need to
stream the whole dataset).

**Output files:**
- `data/libero_subset/data.parquet` — all frames as a table
- `data/libero_subset/task_names.json` — maps task_index → task description string

### `normalize.py`

Reads the Parquet file and computes **mean and standard deviation** for:
- Actions (7-dim)
- States (8-dim)

Normalization is important so that all dimensions are on a similar scale during training.
Must be run once before training.

**Output file:** `data/libero_subset/norm_stats.json`

### `dataset.py`

Defines `LiberoDataset` — a standard PyTorch `Dataset` that:
1. Loads the Parquet file into a DataFrame
2. Groups frames by episode
3. For each frame (sample), loads:
   - Two images (decoded from PNG bytes → float32 tensor)
   - Robot state (float32, normalized)
   - Language tokens (from SmolVLM2 tokenizer, length 48)
   - Action chunk: the next `chunk_size=50` actions (zero-padded at end of episode)
   - `action_is_pad`: boolean mask marking padded positions

Also provides `make_splits()` which splits episodes into train/val sets.

### `normalize.py`

Simple script that reads the Parquet file and computes mean/std statistics for
actions and states. Run once before training.

### `train.py`

The main training loop. Key steps:

1. **Load norm stats** from `norm_stats.json`
2. **Build datasets** using `make_splits()`
3. **Build model** — either load pretrained `lerobot/smolvla_base` or build from config
4. **Freeze layers** — by default only action expert layers are trainable
5. **Create optimizer** — AdamW with (β1=0.9, β2=0.95)
6. **Create scheduler** — linear warmup then cosine decay
7. **Training loop** — infinite data iterator, gradient clipping, periodic logging,
   validation, and checkpointing
8. **Save final checkpoint**

---

## 11. Training Arguments Reference

| Argument | Default | Description |
|---|---|---|
| `--data` | `data/libero_subset` | Path to downloaded LIBERO directory |
| `--output` | `checkpoints/run1` | Where to save checkpoints |
| `--from_pretrained` | `lerobot/smolvla_base` | Pretrained model (HF repo or local dir). Set to `''` for random init |
| `--steps` | `5000` | Total number of gradient steps |
| `--batch` | `4` | Batch size per step |
| `--lr` | `1e-4` | Peak learning rate |
| `--weight_decay` | `1e-10` | AdamW weight decay |
| `--grad_clip` | `10.0` | Gradient clipping norm |
| `--warmup_steps` | `200` | Steps for linear LR warmup |
| `--chunk` | `50` | Action chunk size (50 steps = 5 s at 10 Hz) |
| `--device` | `cuda` or `cpu` | Device to train on |
| `--seed` | `42` | Random seed |
| `--num_workers` | `2` | DataLoader worker processes |
| `--log_every` | `50` | Log loss every N steps |
| `--save_every` | `500` | Save checkpoint every N steps |
| `--val_every` | `500` | Run validation every N steps |
| `--val_fraction` | `0.1` | Fraction of episodes to hold out for validation |
| `--train_expert_only` | `True` | Freeze VLM, only train action expert (recommended) |

---

## 12. GPU Memory Guide

| Configuration | Approx VRAM |
|---|---|
| `--train_expert_only`, batch 4 | ~16 GB |
| `--train_expert_only`, batch 2 | ~10 GB |
| `--train_expert_only`, batch 1 | ~7 GB |
| Full model (no freeze), batch 2 | ~24 GB |
| Full model (no freeze), batch 1 | ~14 GB |

If you run out of memory:
- Reduce `--batch` to 2 or 1
- Use `--train_expert_only` (default, already freezes VLM)
- Use `--device cpu` for testing (slow but works)

---

## 13. What Gets Trained vs Frozen

With `--train_expert_only` (default):

```
policy
└── model (VLAFlowMatching)
    ├── vlm_with_expert
    │   ├── vlm              ← FROZEN (SmolVLM2 backbone)
    │   │   ├── vision_model ← FROZEN (SigLIP)
    │   │   └── text_model   ← FROZEN (SmolLM2, 16 layers)
    │   └── lm_expert        ← TRAINED (Action Expert, 16 layers)
    ├── state_proj           ← TRAINED (Linear: state → VLM token)
    ├── action_in_proj       ← TRAINED (Linear: noisy actions → tokens)
    ├── action_out_proj      ← TRAINED (Linear: expert hidden → actions)
    ├── action_time_mlp_in   ← TRAINED (MLP for time embedding)
    └── action_time_mlp_out  ← TRAINED (MLP for time embedding)
```

The reasoning: the VLM already has excellent visual and language understanding from
pretraining. The action expert is what needs to learn LIBERO-specific manipulation skills.
Freezing the VLM also saves significant GPU memory and speeds up training.

---

## 14. Batch Format SmolVLA Expects

`SmolVLAPolicy.forward(batch)` expects a dictionary with these exact keys:

```python
batch = {
    # Images: float32 tensors in [0, 1], shape (B, 3, H, W)
    "observation.images.image":            torch.Tensor,   # (B, 3, 256, 256)
    "observation.images.image2":           torch.Tensor,   # (B, 3, 256, 256)

    # Robot state: shape (B, state_dim)
    "observation.state":                   torch.Tensor,   # (B, 8)

    # Language tokens from SmolVLM2 tokenizer, max_length=48
    "observation.language_tokens":         torch.LongTensor,  # (B, 48)
    "observation.language_attention_mask": torch.LongTensor,  # (B, 48)

    # Future actions for supervised training loss
    "action":                              torch.Tensor,      # (B, 50, 7) — normalized
    "action_is_pad":                       torch.BoolTensor,  # (B, 50)    — True = padded
}
```

**Returns:** `(loss_scalar, loss_dict)` — call `loss.backward()` directly.

**Important notes:**
- Images must be `float32` in `[0, 1]`. SmolVLA rescales internally.
- Actions and states must be **MEAN_STD normalized** using the stats from `norm_stats.json`.
- Language descriptions must end with `\n`.
- `action_is_pad=True` means that timestep is padding (end of episode) and is excluded from loss.

---

## 15. Bugs Fixed During Debugging

These bugs were discovered when running the scripts end-to-end and have already been
patched in the project files.

| # | File | Error | Fix |
|---|---|---|---|
| 1 | `normalize.py` | `NameError: Optional not defined` | Moved `from typing import Optional` to top of file |
| 2 | `modeling_smolvla.py` | `ImportError: cannot import 'Unpack' from 'typing'` | Python 3.10 doesn't have `Unpack` in `typing` (added in 3.11). Added try/except fallback to `typing_extensions` |
| 3 | `configs/policies.py` | `AttributeError: 'super' object has no attribute '__post_init__'` | `SmolVLAConfig.__post_init__` calls `super().__post_init__()` but `PreTrainedConfig` had none. Added `def __post_init__(self): pass` |
| 4 | `configs/policies.py` | `AttributeError: 'SmolVLAConfig' has no attribute 'device'` | Several fields (`device`, `use_amp`, `push_to_hub`, etc.) live on the real `PreTrainedConfig` base class. Added them to our stub. |
| 5 | pip package | `ImportError: Package num2words required` | SmolVLM2 processor needs `num2words`. Added `pip install num2words` to `setup.sh` |
| 6 | `train.py` | `ValueError: optimizer got an empty parameter list` | Parameter names are under `model.*` (e.g. `model.lm_expert.*`) not bare names. Fixed prefix matching in `build_model()` |
| 7 | `configs/policies.py` | `AttributeError: 'SmolVLAConfig' has no attribute 'image_features'` | `image_features`, `action_feature`, `state_feature` are `@property` methods on the real base class. Added them to `PreTrainedConfig` stub |
| 8 | `modeling_smolvla.py` | `RuntimeError: where expected boolean tensor, got dtype Long` | In `make_att_2d_masks`, multiplying two int64 pad masks produces int64 not bool. Added `.bool()` cast on the result |

---

## 16. Checkpoints and Resuming

Checkpoints are saved as:
```
checkpoints/run1/
├── step_0000500/
│   ├── model.safetensors    ← model weights
│   └── config.json          ← SmolVLAConfig
├── step_0001000/
│   └── ...
└── final/
    ├── model.safetensors
    └── config.json
```

To resume training from a checkpoint, pass it as `--from_pretrained`:

```bash
python train.py \
    --from_pretrained checkpoints/run1/step_0001000 \
    --steps 10000 \
    --output checkpoints/run1_continued
```

To use a finetuned checkpoint for inference:

```python
import sys
sys.path.insert(0, "/path/to/smolvla_finetune")

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

policy = SmolVLAPolicy.from_pretrained("checkpoints/run1/final")
policy.eval()

# Build your batch dict and call:
with torch.no_grad():
    actions = policy.select_action(batch)
```

---

*Generated with Claude Code — see `train.py`, `dataset.py`, and the `lerobot/` stubs for full implementation details.*
