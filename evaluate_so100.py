"""
evaluate_so100.py
=================
Evaluate a SmolVLAPolicy checkpoint on the SO-100 pick-and-place dataset.

Run this BEFORE fine-tuning to get the baseline, and AFTER to compare.

Metrics:
  flow_loss  — flow-matching MSE loss (same as training validation)
  action_mae — mean absolute error of predicted actions per dimension
               (in normalized units; lower is better)

Usage:
    # Step 1 — baseline (pretrained, before fine-tuning)
    python evaluate_so100.py \
        --checkpoint lerobot/smolvla_base \
        --data       data/so100_pickplace \
        --out        results/so100_pretrained.json

    # Step 2 — after fine-tuning
    python evaluate_so100.py \
        --checkpoint checkpoints/so100_run/final \
        --data       data/so100_pickplace \
        --out        results/so100_finetuned.json

    # Step 3 — compare
    python compare_so100.py results/so100_pretrained.json results/so100_finetuned.json
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import FeatureType, PolicyFeature
from dataset_so100 import SO100Dataset, make_splits
from model import SmolVLAPolicy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("evaluate_so100")


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate SmolVLA on SO-100 pick-and-place"
    )
    p.add_argument("--checkpoint",    required=True,
                   help="HF repo ID (e.g. lerobot/smolvla_base) or "
                        "local dir (e.g. checkpoints/so100_run/final)")
    p.add_argument("--data",          default="data/so100_pickplace")
    p.add_argument("--split",         default="val",
                   choices=["val", "train", "all"])
    p.add_argument("--val_fraction",  type=float, default=0.1)
    p.add_argument("--batch",         type=int,   default=4)
    p.add_argument("--chunk",         type=int,   default=50)
    p.add_argument("--num_workers",   type=int,   default=2)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--no_action_mae", action="store_true",
                   help="Skip the slower action-MAE metric")
    p.add_argument("--max_per_task",  type=int,   default=None,
                   help="Sample at most N frames per task (faster eval)")
    p.add_argument("--subset_file",   default="results/so100_eval_subset.json",
                   help="Path to save/load sampled frame indices for "
                        "reproducibility across pre/post finetune runs")
    p.add_argument("--out",           default=None,
                   help="Path to save results JSON "
                        "(e.g. results/so100_pretrained.json)")
    return p.parse_args()


# ── model loading ──────────────────────────────────────────────────────────

def load_policy(checkpoint: str, action_dim: int, state_dim: int,
                device: str, chunk_size: int = 50) -> SmolVLAPolicy:
    log.info(f"Loading checkpoint: {checkpoint}")
    policy = SmolVLAPolicy.from_pretrained(checkpoint)

    # Patch feature specs to SO-100 dims
    policy.config.input_features = {
        "observation.state": PolicyFeature(
            type=FeatureType.STATE, shape=(state_dim,)
        ),
        "observation.images.image": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 256, 256)
        ),
        "observation.images.image2": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 256, 256)
        ),
    }
    policy.config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))
    }
    # Sync chunk_size so embed_suffix att_masks length matches the DataLoader
    policy.config.chunk_size     = chunk_size
    policy.config.n_action_steps = chunk_size

    policy = policy.to(device)
    policy.eval()

    n_total = sum(p.numel() for p in policy.parameters())
    log.info(f"Model loaded: {n_total/1e6:.1f}M parameters  (chunk_size={chunk_size})")
    return policy


# ── metrics ────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_flow_loss(policy, loader, device) -> float:
    total, n = 0.0, 0
    for batch in tqdm(loader, desc="flow_loss", leave=False):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        loss, _ = policy(batch)
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def compute_action_mae(policy, loader, device, action_dim: int) -> dict:
    """Full denoising (10 Euler steps) → MAE per action dimension."""
    sum_abs_err = torch.zeros(action_dim, device=device)
    count        = torch.tensor(0.0, device=device)

    for batch in tqdm(loader, desc="action_mae (denoising)", leave=False):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        images, img_masks = policy.prepare_images(batch)
        state             = policy.prepare_state(batch)
        lang_tokens       = batch["observation.language_tokens"]
        lang_masks        = batch["observation.language_attention_mask"]

        # Full denoising → (B, chunk_size, max_action_dim)
        pred = policy.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state
        )
        pred = pred[:, :, :action_dim]          # trim padding

        gt     = batch["action"]                # (B, chunk, action_dim)
        is_pad = batch["action_is_pad"]         # (B, chunk)
        mask   = (~is_pad).float()              # (B, chunk)

        abs_err = (pred - gt).abs()             # (B, chunk, action_dim)
        abs_err = abs_err * mask.unsqueeze(-1)

        sum_abs_err += abs_err.sum(dim=(0, 1))
        count        += mask.sum()

    if count == 0:
        return {"per_dim_mae": [0.0] * action_dim, "mean_mae": 0.0}

    per_dim = (sum_abs_err / count).cpu().tolist()
    return {
        "per_dim_mae": per_dim,
        "mean_mae":    float(np.mean(per_dim)),
    }


# ── subsample ──────────────────────────────────────────────────────────────

def subsample_per_task(dataset: SO100Dataset, max_per_task: int,
                       subset_file: str, seed: int) -> SO100Dataset:
    """Subsample ≤ max_per_task frames per task; save/load indices for reproducibility."""
    import copy

    subset_path = Path(subset_file)

    if subset_path.exists():
        log.info(f"Loading existing subset indices from {subset_path}")
        saved = json.load(open(subset_path))
        index_set = {(ep, li) for ep, li in saved["index"]}
        new_ds = copy.copy(dataset)
        new_ds._index = [
            (ep, li) for ep, li in dataset._index if (ep, li) in index_set
        ]
        return new_ds

    rng = np.random.default_rng(seed)
    task_to_indices: dict[int, list] = {}
    for ep, local_i in dataset._index:
        global_row = dataset._episodes[ep][local_i]
        task_idx   = int(dataset.df.iloc[global_row]["task_index"])
        task_to_indices.setdefault(task_idx, []).append((ep, local_i))

    sampled = []
    for task_idx, indices in sorted(task_to_indices.items()):
        if len(indices) > max_per_task:
            chosen = rng.choice(len(indices), max_per_task, replace=False)
            sampled.extend(indices[i] for i in chosen)
        else:
            sampled.extend(indices)

    subset_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"max_per_task": max_per_task, "index": sampled},
              open(subset_path, "w"))
    log.info(f"Saved {len(sampled)} frame indices → {subset_path}")

    new_ds = copy.copy(dataset)
    new_ds._index = sampled
    return new_ds


# ── main ───────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = args.device

    # ── Load SO-100 metadata ──────────────────────────────────────────
    meta_path = Path(args.data) / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"metadata.json not found in {args.data}. "
            "Run download_so100.py first."
        )
    with open(meta_path) as f:
        meta = json.load(f)

    action_dim = meta["action_dim"]
    state_dim  = meta["state_dim"]
    log.info(f"SO-100 config: action_dim={action_dim}, state_dim={state_dim}")

    # ── Norm stats ─────────────────────────────────────────────────────
    norm_path  = Path(args.data) / "norm_stats.json"
    norm_stats = None
    if norm_path.exists():
        with open(norm_path) as f:
            norm_stats = json.load(f)
        log.info(f"Normalization stats loaded from {norm_path}")
    else:
        log.warning("No norm_stats.json — running without normalization")

    # ── Dataset ────────────────────────────────────────────────────────
    if args.split == "all":
        dataset = SO100Dataset(
            args.data, chunk_size=args.chunk, norm_stats=norm_stats
        )
        log.info(f"Evaluating on full dataset: {len(dataset)} samples")
    else:
        train_ds, val_ds = make_splits(
            args.data,
            chunk_size=args.chunk,
            norm_stats=norm_stats,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
        dataset = val_ds if args.split == "val" else train_ds
        log.info(f"Evaluating on {args.split} split: {len(dataset)} samples")

    # Optional per-task subsampling
    if args.max_per_task is not None:
        dataset = subsample_per_task(
            dataset, args.max_per_task, args.subset_file, args.seed
        )
        log.info(
            f"Subsampled to {len(dataset)} frames "
            f"(≤{args.max_per_task} per task)"
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    # ── Model ──────────────────────────────────────────────────────────
    policy = load_policy(
        args.checkpoint, action_dim=action_dim, state_dim=state_dim,
        device=device, chunk_size=args.chunk
    )

    # ── Evaluate ───────────────────────────────────────────────────────
    results = {
        "checkpoint":  args.checkpoint,
        "split":       args.split,
        "n_samples":   len(dataset),
        "action_dim":  action_dim,
        "state_dim":   state_dim,
    }

    log.info("Computing flow-matching loss ...")
    flow_loss = compute_flow_loss(policy, loader, device)
    results["flow_loss"] = flow_loss
    log.info(f"  flow_loss = {flow_loss:.6f}")

    if not args.no_action_mae:
        log.info("Computing action MAE (full denoising — slower) ...")
        mae_results = compute_action_mae(policy, loader, device, action_dim)
        results.update(mae_results)
        log.info(f"  mean_action_mae = {mae_results['mean_mae']:.6f}")
        per_dim_str = ", ".join(f"{v:.4f}" for v in mae_results["per_dim_mae"])
        log.info(f"  per_dim_mae     = [{per_dim_str}]")

    # ── Print summary ──────────────────────────────────────────────────
    # Build per-dim labels: j1..j(N-1), grip
    joint_labels = [f"j{i+1}" for i in range(action_dim - 1)] + ["grip"]

    print("\n" + "=" * 60)
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Split      : {args.split}  ({len(dataset)} samples)")
    print(f"  action_dim : {action_dim}  |  state_dim : {state_dim}")
    print(f"  flow_loss  : {results['flow_loss']:.6f}")
    if "mean_mae" in results:
        print(f"  mean_mae   : {results['mean_mae']:.6f}  (normalized units)")
        for lbl, v in zip(joint_labels, results["per_dim_mae"]):
            print(f"    {lbl:4s}  {v:.4f}")
    print("=" * 60 + "\n")

    # ── Save JSON ──────────────────────────────────────────────────────
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"Results saved to {out_path}")

    return results


if __name__ == "__main__":
    main()
