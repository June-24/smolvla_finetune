# SmolVLA Finetune — Codebase Guide

This project finetunes **SmolVLA** (a Vision-Language-Action model) on the **LIBERO** robot manipulation dataset — entirely without lerobot as a dependency. Everything runs as plain PyTorch + HuggingFace Transformers.

---

## What is SmolVLA?

SmolVLA is a robot policy model. It takes:
- **Two camera images** (wrist + workspace) from the robot
- **Robot joint state** (8 numbers: 7 joints + gripper)
- **A language instruction** (e.g., "pick up the red block")

And outputs:
- **A chunk of 50 future actions** (7 numbers each: 6 joint velocities + gripper)

The model is built on top of **SmolVLM2-500M**, a small Vision-Language Model originally designed for image/text chat. SmolVLA adds an "action expert" transformer on top that cross-attends to the VLM's representations to produce robot actions.

---

## How Finetuning Works (Without lerobot)

The original SmolVLA was released with the lerobot library, which handles data loading, training loops, and checkpointing. This codebase replaces all of that with custom code:

| lerobot component | This codebase |
|---|---|
| `LeRobotDataset` | `dataset.py` → `LiberoDataset` |
| Training loop | `train.py` |
| Config system | `config.py` → `SmolVLAConfig` |
| Evaluation | `evaluate.py` |
| Normalization | `normalize.py` |

The model weights (`model.py`, `expert.py`) are directly ported from lerobot's SmolVLA source with all lerobot imports removed.

---

## File-by-File Explanation

### `config.py` — Configuration

Defines all hyperparameters and feature specifications as a Python dataclass.

**Key concepts:**

- **`FeatureType`** — an enum describing what kind of data each input/output is:
  - `STATE` = robot joint positions (numbers)
  - `VISUAL` = camera image (pixels)
  - `ACTION` = robot actions (numbers)

- **`PolicyFeature`** — pairs a `FeatureType` with a `shape`. For example:
  ```python
  "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,))
  "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))
  "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))
  ```

- **`SmolVLAConfig`** — the main config dataclass. Important fields:
  - `chunk_size=50` — how many future action steps to predict at once
  - `max_state_dim=32` / `max_action_dim=32` — the model always pads inputs to 32 dims internally (so it can handle any robot without architecture changes)
  - `train_expert_only=True` — freeze the entire VLM backbone, only train the action expert (much faster, less GPU memory)
  - `freeze_vision_encoder=True` — additionally freeze the image encoder
  - `attention_mode="cross_attn"` — the action expert cross-attends to VLM key/value caches rather than doing full self-attention over everything
  - `num_steps=10` — number of denoising steps during inference (flow matching)

- **`save_pretrained` / `from_pretrained`** — serialize/deserialize the config to a `config.json` file. Handles enum serialization and the HuggingFace Hub's uppercase `"STATE"` → lowercase `"state"` mismatch.

---

### `expert.py` — The Neural Network Architecture

This is the core architecture file. It defines `SmolVLMWithExpertModel`, which is a combination of two transformers running together.

**The two transformers:**

1. **VLM (SmolVLM2-500M)** — the big pretrained Vision-Language Model. It processes images and language tokens. Its weights are mostly frozen during finetuning.

2. **Action Expert (lm_expert)** — a smaller transformer (75% the hidden size of the VLM) with the same number of layers. It processes the action/state tokens and learns to predict robot actions.

**How they interact — cross-attention:**

At each transformer layer, instead of having the expert do full self-attention over everything, the expert queries **cross-attend to the VLM's key-value cache**. This means:

```
VLM layer i:     processes [image tokens + language tokens] → produces K, V cache
Expert layer i:  processes [action tokens + state tokens]   → Q attends to VLM's K, V
```

This is efficient because the VLM's KV cache only needs to be computed once per inference call (the image/language context doesn't change across denoising steps).

**`apply_rope`** — Rotary Position Encoding. Used to inject position information into attention queries and keys without adding learned embeddings. Standard in modern LLMs.

**`get_intermediate_size`** — Computes the FFN hidden size for the expert following LLaMA-style sizing (2/3 × hidden × multiplier, rounded to multiple of 256).

**`set_requires_grad`** — Controls which parameters are frozen:
- If `train_expert_only=True`: VLM is fully frozen, only the `lm_expert` trains
- Otherwise: Most VLM layers train too, but the last layer and LM head are frozen (to preserve the representation quality)

**`forward`** — The main forward pass runs both transformers in lockstep, layer by layer:
1. Pass image+language embeddings through VLM layer → get K, V
2. Pass action embeddings through expert layer → Q attends to above K, V
3. Apply residual connections and MLP
4. Repeat for all layers
5. Apply final layer norm

---

### `model.py` — Policy + Flow Matching

Wraps the architecture in a complete trainable policy.

**`VLAFlowMatching`** — the main model class. Implements **Conditional Flow Matching** (CFM), a modern alternative to diffusion that's faster and more stable.

**What is Flow Matching?**

Instead of predicting noise like diffusion models, flow matching learns to predict a velocity field that transforms random noise into actions. During training:

1. Sample a random noise action `x_0 ~ N(0, I)`
2. Sample a random timestep `t ~ Uniform(0, 1)`
3. Interpolate: `x_t = (1 - t) * x_0 + t * x_1` where `x_1` is the ground-truth action
4. The model predicts the velocity `v = x_1 - x_0` (direction from noise to real action)
5. Loss = MSE between predicted velocity and true velocity

During inference, start from noise and run 10 Euler steps to reach an action.

**Key components of `VLAFlowMatching`:**

- **`state_proj`** — linear layer that maps the 8-dim robot state to the expert's hidden size (padded to 32 first, then projected)
- **`action_in_proj`** — linear layer that maps the 7-dim noisy action chunk to the expert's hidden size
- **`action_out_proj`** — linear layer that maps expert hidden states back to 32-dim action predictions
- **`action_time_mlp_in/out`** — small MLP that converts the scalar timestep `t` into an embedding added to each action token (so the model knows "how noisy" the current action is)
- **`prepare_images`** — takes raw image tensors `(B, 3, H, W)`, resizes/pads them to 512×512, and runs them through the VLM's processor to get pixel values + attention masks
- **`prepare_state`** — pads 8-dim state → 32-dim with zeros, then projects to expert hidden size

**`forward` (training):**

```
batch → prepare images → embed in VLM
      → prepare state → project to expert
      → sample noise + timestep t
      → interpolate noisy action
      → run both transformers (expert attends to VLM)
      → predict velocity
      → MSE loss vs true velocity
```

**`sample_actions` (inference):**

```
x = random noise (shape: B × chunk_size × action_dim)
for t in linspace(0, 1, num_steps=10):
    v = model(x, t, images, language, state)   # predict velocity
    x = x + v * dt                              # Euler step
return x   # final action chunk
```

**`SmolVLAPolicy`** — thin wrapper around `VLAFlowMatching` that adds `save_pretrained` / `from_pretrained` (downloads from HuggingFace Hub or loads from local directory) and `select_action` (for real-time robot deployment with a rolling observation queue).

---

### `dataset.py` — Data Loading

**`LiberoDataset`** — PyTorch Dataset that reads from a Parquet file.

**The data format:**

Each row in the Parquet file is one timestep from a robot episode:
- `episode_index` — which episode (trajectory) this frame belongs to
- `frame_index` — position within the episode
- `task_index` — which of the 40 LIBERO tasks
- `action` — 7-float list: the action taken at this step
- `observation.state` — 8-float list: joint positions at this step
- `observation.images.image` — PNG bytes: workspace camera image
- `observation.images.image2` — PNG bytes: wrist camera image

**Memory-efficient loading — the key challenge:**

The Parquet file is ~30 GB because it stores ~270k full PNG images. Loading everything into pandas would immediately OOM. The solution is split loading:

1. **Small columns only → pandas**: episode index, frame index, task, action, state (~tens of MB total). This is what's used for building the dataset index and action chunks.

2. **Images → lazy row-group loading**: Instead of loading image bytes into RAM, a `ParquetFile` is opened once per DataLoader worker, and only the specific **row group** containing the requested row is read from disk. One row group is cached — if the next sample is in the same group, no disk I/O occurs.

```
__getitem__(idx):
    ep_id, local_i = self._index[idx]
    global_row = self._episodes[ep_id][local_i]

    # Fast: from pandas (already in RAM)
    row = self.df.iloc[global_row]
    state, action = row["observation.state"], row["action"]

    # Lazy: find the row group, read it (or use cache), extract bytes
    rg_idx, local_row = find_row_group(global_row)
    img1, img2 = self._cached_rg[rg_idx][local_row]
```

**Action chunks:**

The model doesn't predict just one action — it predicts the next 50 actions at once (`chunk_size=50`). For each timestep `t`, the dataset collects actions from `t` to `t+49`. If the episode ends before 50 steps are available, remaining slots are zero-padded and marked with `action_is_pad=True`. The loss ignores padded steps.

**Normalization:**

Actions and states are normalized to zero mean / unit variance using stats computed by `normalize.py`. Images are left as-is (divided by 255 to get [0, 1] floats).

**`make_splits`** — splits episodes (not frames) into train/val sets. Splitting by episode prevents data leakage where frame `t` is in train and frame `t+1` is in val.

---

### `normalize.py` — Normalization Stats

Reads only the `action` and `observation.state` columns from the Parquet file (skipping image bytes) and computes per-dimension mean and standard deviation. Saves to `norm_stats.json`:

```json
{
  "action": {"mean": [0.01, -0.02, ...], "std": [0.15, 0.23, ...]},
  "state":  {"mean": [0.0, 1.57, ...],   "std": [0.12, 0.08, ...]}
}
```

Must be run once before training.

---

### `train.py` — Training Loop

**`build_model`** — loads `SmolVLAPolicy` from `lerobot/smolvla_base` on HuggingFace, then patches the config with LIBERO-specific feature shapes. Freezes the VLM backbone and only enables gradients for:
- `model.vlm_with_expert.lm_expert` — the action expert transformer
- `model.state_proj` — state projection layer
- `model.action_in_proj` / `model.action_out_proj` — action input/output projections
- `model.action_time_mlp_in` / `model.action_time_mlp_out` — timestep embedding MLP

This means only ~20-30% of the model's parameters are actually trained.

**`WarmupCosineSchedule`** — custom LR scheduler (no external dependencies):
- First `warmup_steps` (200): LR linearly increases from 0 to `lr`
- After that: LR follows a cosine curve down to 10% of `lr`

**Training loop:**

```
for step in range(total_steps):
    batch = next(infinite_data_iterator)
    loss, _ = policy(batch)         # flow matching forward pass
    loss.backward()
    clip_grad_norm_(params, 10.0)   # gradient clipping for stability
    optimizer.step()
    scheduler.step()

    if step % 500 == 0:
        save checkpoint
        compute val loss
```

The data iterator loops infinitely — it restarts the DataLoader when the dataset is exhausted, which handles the case where `steps > len(dataset) / batch_size`.

Checkpoints are saved as HuggingFace-compatible directories (`model.safetensors` + `config.json`) every 500 steps, with a final save at the end.

---

### `evaluate.py` — Evaluation

Computes two metrics on a held-out subset:

**1. `flow_loss`** — runs the training forward pass (with random timestep `t`) on each batch and averages the MSE loss. Fast. Same number as the training validation loss.

**2. `action_mae`** (optional, slower) — runs full denoising inference (10 Euler steps) to get actual predicted actions, then computes Mean Absolute Error vs ground-truth actions. Reported in normalized units (0 = perfect, 1 = off by 1 standard deviation).

**`--max_per_task 100`** — subsamples 100 frames per task (100 × 40 tasks = ~4000 samples) so evaluation is fast. The sampled indices are saved to `results/eval_subset.json` on the first run and reloaded on subsequent runs, ensuring the pre-finetune and post-finetune evaluations are compared on **exactly the same frames**.

---

### `merge_chunks.py` — Merge Dataset Chunks

The full LIBERO dataset was downloaded in 4 chunks (0-9, 10-19, 20-29, 30-39 tasks each). This script merges them into a single `data/libero_full/data.parquet` file using **row-group streaming**: it reads one row group at a time from each input file and writes it directly to the output, so RAM usage stays at ~1 GB regardless of total dataset size.

---

## Data Flow Diagram

```
data/chunk_0to9/data.parquet   ─┐
data/chunk_10to19/data.parquet  ├─ merge_chunks.py ──► data/libero_full/data.parquet
data/chunk_20to29/data.parquet  │
data/chunk_30to39/data.parquet ─┘

data/libero_full/data.parquet ──► normalize.py ──► norm_stats.json

                              ┌── norm_stats.json
evaluate.py (pretrained) ◄───┤   data/chunk_0to9/data.parquet      ► results/pretrained.json
                              └── checkpoint: lerobot/smolvla_base

                              ┌── norm_stats.json
train.py ◄───────────────────┤   data/chunk_0to9/data.parquet
                              └── checkpoint: lerobot/smolvla_base
                                                  │
                                                  ▼
                                     checkpoints/run_chunk0to9/final

                              ┌── norm_stats.json
evaluate.py (finetuned) ◄────┤   data/chunk_0to9/data.parquet      ► results/finetuned.json
                              └── checkpoint: checkpoints/run_chunk0to9/final
```

---

## Full Workflow Commands

```bash
conda activate smolvla
cd '/mnt/c/Users/Moham/OneDrive/Desktop/Claude Projects/smolvla_finetune'

# 1. Compute normalization stats
python normalize.py --data data/chunk_0to9

# 2. Evaluate pretrained model (saves subset indices to results/eval_subset.json)
python evaluate.py --checkpoint lerobot/smolvla_base --data data/chunk_0to9 --split all --max_per_task 100 --out results/pretrained.json

# 3. Finetune on the chunk (adjust --batch if you hit OOM)
python train.py --data data/chunk_0to9 --output checkpoints/run_chunk0to9 --from_pretrained lerobot/smolvla_base --steps 5000 --batch 4 --lr 1e-4 --device cuda

# 4. Evaluate finetuned model (reloads same subset indices → fair comparison)
python evaluate.py --checkpoint checkpoints/run_chunk0to9/final --data data/chunk_0to9 --split all --max_per_task 100 --out results/finetuned.json

# 5. Compare
python -c "
import json
pre = json.load(open('results/pretrained.json'))
ft  = json.load(open('results/finetuned.json'))
print(f'flow_loss: {pre[\"flow_loss\"]:.6f}  ->  {ft[\"flow_loss\"]:.6f}')
if 'mean_mae' in pre:
    print(f'mean_mae:  {pre[\"mean_mae\"]:.6f}  ->  {ft[\"mean_mae\"]:.6f}')
"
```

---

## Why No lerobot?

lerobot is a large library with many dependencies (gym, gymnasium, datasets, etc.) and assumes a specific project structure. This codebase strips it down to the essentials:

- No gym/gymnasium — we evaluate on offline data, not a simulator
- No `datasets` library — we read Parquet directly with pyarrow
- No lerobot config system — we use plain Python dataclasses
- No lerobot training infra — we use a plain PyTorch loop
- Model weights are still loaded from `lerobot/smolvla_base` on HuggingFace — we just don't need the library to use them

The model code (`model.py`, `expert.py`) is a direct port of lerobot's `modeling_smolvla.py` and `smolvlm_with_expert.py` with all lerobot imports replaced by inline implementations.
