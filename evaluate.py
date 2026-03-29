"""
evaluate.py
===========
Evaluate a SmolVLAPolicy checkpoint on the LIBERO validation set.

Reports two metrics:
  1. flow_loss  – flow-matching MSE loss (same as training validation metric)
  2. action_mae – mean absolute error of predicted actions per dimension
                  (normalized units: 0 = perfect, 1 = off by 1 std dev)

Usage
-----
# Baseline: pretrained model before any finetuning
python evaluate.py --checkpoint lerobot/smolvla_base --data data/libero_subset

# Finetuned: run1 final checkpoint
python evaluate.py --checkpoint checkpoints/run1/final --data data/libero_subset

# Full dataset instead of only the val split
python evaluate.py --checkpoint checkpoints/run1/final --split all

# Save results to a JSON file for easy comparison
python evaluate.py --checkpoint lerobot/smolvla_base   --out results/pretrained.json
python evaluate.py --checkpoint checkpoints/run1/final --out results/finetuned.json
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import FeatureType, PolicyFeature
from dataset import LiberoDataset, make_splits
from model import SmolVLAPolicy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("evaluate")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate SmolVLA on LIBERO")
    p.add_argument("--checkpoint",   required=True,
                   help="HF repo ID (e.g. lerobot/smolvla_base) or local dir "
                        "(e.g. checkpoints/run1/final)")
    p.add_argument("--data",         default="data/libero_subset")
    p.add_argument("--split",        default="val",
                   choices=["val", "train", "all"],
                   help="Which subset to evaluate on")
    p.add_argument("--val_fraction", type=float, default=0.1,
                   help="Fraction used as validation split (must match training)")
    p.add_argument("--batch",        type=int, default=4)
    p.add_argument("--chunk",        type=int, default=50)
    p.add_argument("--num_workers",  type=int, default=2)
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--no_action_mae", action="store_true",
                   help="Skip the slower action-MAE metric (full denoising)")
    p.add_argument("--out",          default=None,
                   help="Optional path to save results JSON (e.g. results/pretrained.json)")
    p.add_argument("--max_per_task", type=int, default=None,
                   help="Sample at most N frames per task (e.g. 100). "
                        "Indices are saved/loaded via --subset_file so both "
                        "pre- and post-finetune runs use the exact same frames.")
    p.add_argument("--subset_file",  default="results/eval_subset.json",
                   help="Path to save/load sampled frame indices for reproducibility")
    return p.parse_args()


# ── model loading ──────────────────────────────────────────────────────────────

def load_policy(checkpoint: str, device: str):
    log.info(f"Loading checkpoint: {checkpoint}")
    policy = SmolVLAPolicy.from_pretrained(checkpoint)

    # Ensure LIBERO feature spec is set (shapes used by prepare_images / prepare_state)
    input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        "observation.images.image":  PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
    }
    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))
    }
    policy.config.input_features  = input_features
    policy.config.output_features = output_features

    policy = policy.to(device)
    policy.eval()

    n_total = sum(p.numel() for p in policy.parameters())
    log.info(f"Model loaded: {n_total / 1e6:.1f}M parameters")
    return policy


# ── metrics ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_flow_loss(policy, loader, device) -> float:
    """Average flow-matching loss over the dataset (same as training val metric)."""
    total, n = 0.0, 0
    for batch in tqdm(loader, desc="flow_loss", leave=False):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        loss, _ = policy(batch)
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def compute_action_mae(policy, loader, device) -> dict:
    """
    Full denoising inference → MAE per action dimension (normalized units).

    Returns
    -------
    dict with keys:
        "per_dim_mae"   – list of 7 floats (one per action DOF)
        "mean_mae"      – scalar mean over all dims
    """
    action_dim = policy.config.action_feature.shape[0]
    sum_abs_err = torch.zeros(action_dim, device=device)
    count        = torch.tensor(0.0, device=device)

    for batch in tqdm(loader, desc="action_mae (denoising)", leave=False):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Prepare inputs the same way predict_action_chunk would
        images, img_masks = policy.prepare_images(batch)
        state             = policy.prepare_state(batch)
        lang_tokens  = batch["observation.language_tokens"]
        lang_masks   = batch["observation.language_attention_mask"]

        # Full denoising (10 Euler steps) → (B, chunk_size, max_action_dim)
        pred = policy.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state
        )
        pred = pred[:, :, :action_dim]               # (B, chunk, 7) — trim padding

        gt      = batch["action"]                    # (B, chunk, 7) normalized
        is_pad  = batch["action_is_pad"]             # (B, chunk)
        mask    = (~is_pad).float()                  # (B, chunk)

        # Masked MAE per action dimension
        abs_err = (pred - gt).abs()                  # (B, chunk, 7)
        abs_err = abs_err * mask.unsqueeze(-1)       # zero-out padded steps

        sum_abs_err += abs_err.sum(dim=(0, 1))
        count        += mask.sum()

    if count == 0:
        return {"per_dim_mae": [0.0] * action_dim, "mean_mae": 0.0}

    per_dim = (sum_abs_err / count).cpu().tolist()
    return {
        "per_dim_mae": per_dim,
        "mean_mae": float(np.mean(per_dim)),
    }


# ── subsample ──────────────────────────────────────────────────────────────────

def subsample_per_task(dataset, max_per_task: int, subset_file: str, seed: int):
    """Return a dataset copy whose _index is limited to max_per_task frames per task.

    If subset_file already exists, load indices from it (so pre- and
    post-finetune runs evaluate on the identical frames).  Otherwise sample
    randomly and save indices for future runs.
    """
    import copy
    subset_path = Path(subset_file)

    if subset_path.exists():
        log.info(f"Loading existing subset indices from {subset_path}")
        saved = json.load(open(subset_path))
        index_set = {(ep, li) for ep, li in saved["index"]}
        new_ds = copy.copy(dataset)
        new_ds._index = [(ep, li) for ep, li in dataset._index if (ep, li) in index_set]
        return new_ds

    # Group dataset indices by task_index
    rng = np.random.default_rng(seed)
    task_to_indices: dict[int, list] = {}
    for pos, (ep, local_i) in enumerate(dataset._index):
        global_row = dataset._episodes[ep][local_i]
        task_idx = int(dataset.df.iloc[global_row]["task_index"])
        task_to_indices.setdefault(task_idx, []).append((ep, local_i))

    sampled = []
    for task_idx, indices in sorted(task_to_indices.items()):
        if len(indices) > max_per_task:
            chosen = rng.choice(len(indices), max_per_task, replace=False)
            sampled.extend(indices[i] for i in chosen)
        else:
            sampled.extend(indices)

    # Save for reproducibility
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"max_per_task": max_per_task, "index": sampled}, open(subset_path, "w"))

    new_ds = copy.copy(dataset)
    new_ds._index = sampled
    return new_ds


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = args.device

    # ── load norm stats ──────────────────────────────────────────────────────
    norm_path = Path(args.data) / "norm_stats.json"
    norm_stats = None
    if norm_path.exists():
        with open(norm_path) as f:
            norm_stats = json.load(f)
        log.info(f"Normalization stats loaded from {norm_path}")
    else:
        log.warning("No norm_stats.json found — running without normalization")

    # ── dataset ──────────────────────────────────────────────────────────────
    if args.split == "all":
        dataset = LiberoDataset(args.data, chunk_size=args.chunk, norm_stats=norm_stats)
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

    # ── optional per-task subsample ──────────────────────────────────────────
    if args.max_per_task is not None:
        dataset = subsample_per_task(
            dataset, args.max_per_task, args.subset_file, args.seed
        )
        log.info(f"Subsampled to {len(dataset)} frames "
                 f"(≤{args.max_per_task} per task) — indices in {args.subset_file}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    # ── model ────────────────────────────────────────────────────────────────
    policy = load_policy(args.checkpoint, device)

    # ── evaluate ─────────────────────────────────────────────────────────────
    results = {
        "checkpoint": args.checkpoint,
        "split":      args.split,
        "n_samples":  len(dataset),
    }

    log.info("Computing flow-matching loss ...")
    flow_loss = compute_flow_loss(policy, loader, device)
    results["flow_loss"] = flow_loss
    log.info(f"  flow_loss = {flow_loss:.6f}")

    if not args.no_action_mae:
        log.info("Computing action MAE (full denoising — this is slower) ...")
        mae_results = compute_action_mae(policy, loader, device)
        results.update(mae_results)
        log.info(f"  mean_action_mae = {mae_results['mean_mae']:.6f}")
        per_dim_str = ", ".join(f"{v:.4f}" for v in mae_results["per_dim_mae"])
        log.info(f"  per_dim_mae     = [{per_dim_str}]")

    # ── print summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Split      : {args.split}  ({len(dataset)} samples)")
    print(f"  flow_loss  : {results['flow_loss']:.6f}")
    if "mean_mae" in results:
        print(f"  mean_mae   : {results['mean_mae']:.6f}  (normalized units)")
        labels = ["j1", "j2", "j3", "j4", "j5", "j6", "grip"]
        for lbl, v in zip(labels, results["per_dim_mae"]):
            print(f"    {lbl:4s}  {v:.4f}")
    print("=" * 60 + "\n")

    # ── save JSON ────────────────────────────────────────────────────────────
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"Results saved to {out_path}")

    return results


if __name__ == "__main__":
    main()
