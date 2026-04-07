"""
compare_models_so100.py
=======================
Load and evaluate two SmolVLA checkpoints side-by-side on the SO-100
pick-and-place dataset, then print a comprehensive comparison table.

Standard metrics
----------------
  flow_loss         — flow-matching MSE loss (training validation metric)
  mean_mae          — mean absolute error of denoised actions     (lower = better)
  mean_mse          — mean squared error of denoised actions      (lower = better)

Non-standard metrics (beyond MAE / MSE)
----------------------------------------
  median_mae             — median per-episode MAE; robust to outlier episodes
  p90_mae                — 90th-percentile per-episode MAE; tail / worst-case perf
  success_rate           — % of episodes where mean MAE < --success_thresh
                           (offline proxy for task success rate)
  avg_steps_before_fail  — within each action chunk, how many steps until
                           per-step error > threshold? (offline "steps to completion")
  directional_accuracy   — % of (step, dim) pairs where sign(pred)==sign(gt)
                           i.e. does the model move each joint in the RIGHT direction?
  gripper_accuracy       — binary accuracy on the last action dim (open vs close)
  smoothness_ratio       — mean(|pred_delta|) / mean(|gt_delta|).
                           1.0 matches demo smoothness; >1 jerky; <1 lazy/static
  temporal_drift_slope   — linear slope of per-step MAE across the 50-step chunk.
                           positive = errors grow as the model plans further ahead
  action_magnitude_bias  — per-dim ratio mean(|pred|)/mean(|gt|).
                           <1 means the model underestimates how far to move
                           (action averaging / regression-to-mean, common in BC)

Usage
-----
    # Option A — run full inference on both models (needs GPU)
    python compare_models_so100.py \\
        --baseline   lerobot/smolvla_base \\
        --finetuned  checkpoints/so100_run/final \\
        --data       data/so100_pickplace \\
        --max_per_task 200 \\
        --out        results/full_comparison_so100.json

    # Option B — use pre-computed evaluate_so100.py JSONs (instant, no GPU)
    #            Non-standard metrics will show n/a in this mode.
    python compare_models_so100.py \\
        --baseline_json  results/so100_pretrained.json \\
        --finetuned_json results/so100_finetuned.json
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
from dataset_so100 import SO100Dataset, make_splits
from model import SmolVLAPolicy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("compare_models")

_NAN = float("nan")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare pretrained vs finetuned SmolVLA on SO-100"
    )

    # Option A — run inference
    p.add_argument("--baseline",
                   default=None,
                   help="Baseline checkpoint (HF repo ID or local dir). "
                        "E.g. lerobot/smolvla_base")
    p.add_argument("--finetuned",
                   default=None,
                   help="Finetuned checkpoint. E.g. checkpoints/so100_run/final")

    # Option B — load pre-computed JSON files
    p.add_argument("--baseline_json",  default=None,
                   help="Pre-computed baseline results JSON (from evaluate_so100.py).")
    p.add_argument("--finetuned_json", default=None,
                   help="Pre-computed finetuned results JSON.")

    # Data / eval settings (Option A only)
    p.add_argument("--data",           default="data/so100_pickplace")
    p.add_argument("--split",          default="val",
                   choices=["val", "train", "all"])
    p.add_argument("--val_fraction",   type=float, default=0.1)
    p.add_argument("--batch",          type=int,   default=4)
    p.add_argument("--chunk",          type=int,   default=50)
    p.add_argument("--num_workers",    type=int,   default=2)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--max_per_task",   type=int,   default=None,
                   help="Cap samples per task for faster evaluation.")
    p.add_argument("--subset_file",
                   default="results/so100_eval_subset.json",
                   help="Save/load frame indices so both models see identical data.")
    p.add_argument("--success_thresh", type=float, default=0.5,
                   help="Per-episode mean-MAE threshold that counts as 'success'. "
                        "Also the per-step threshold for avg_steps_before_fail. "
                        "(normalized units, default 0.5)")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out",
                   default="results/full_comparison_so100.json",
                   help="Output JSON path.")
    return p.parse_args()


# ── model / dataset helpers ───────────────────────────────────────────────────

def load_policy(checkpoint: str, action_dim: int, state_dim: int,
                device: str, chunk_size: int = 50) -> SmolVLAPolicy:
    log.info(f"Loading: {checkpoint}")
    policy = SmolVLAPolicy.from_pretrained(checkpoint)
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
    # Must match the DataLoader chunk size — embed_suffix builds att_masks
    # with length self.config.chunk_size, and a mismatch with the actual
    # action tensor length causes a RuntimeError in make_att_2d_masks.
    policy.config.chunk_size     = chunk_size
    policy.config.n_action_steps = chunk_size
    policy = policy.to(device).eval()
    n = sum(par.numel() for par in policy.parameters())
    log.info(f"  {n / 1e6:.1f}M parameters  (chunk_size={chunk_size})")
    return policy


def build_dataset(args, norm_stats) -> SO100Dataset:
    if args.split == "all":
        return SO100Dataset(args.data, chunk_size=args.chunk,
                            norm_stats=norm_stats)
    train_ds, val_ds = make_splits(
        args.data,
        chunk_size=args.chunk,
        norm_stats=norm_stats,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    return val_ds if args.split == "val" else train_ds


def subsample(dataset: SO100Dataset, max_per_task: int,
              subset_file: str, seed: int) -> SO100Dataset:
    import copy
    subset_path = Path(subset_file)
    if subset_path.exists():
        log.info(f"Loading subset indices from {subset_path}")
        saved     = json.load(open(subset_path))
        index_set = {(ep, li) for ep, li in saved["index"]}
        new_ds    = copy.copy(dataset)
        new_ds._index = [(ep, li) for ep, li in dataset._index
                         if (ep, li) in index_set]
        return new_ds

    rng = np.random.default_rng(seed)
    task_to_idx: dict[int, list] = {}
    for ep, li in dataset._index:
        row  = dataset._episodes[ep][li]
        task = int(dataset.df.iloc[row]["task_index"])
        task_to_idx.setdefault(task, []).append((ep, li))

    sampled = []
    for task, indices in sorted(task_to_idx.items()):
        if len(indices) > max_per_task:
            chosen = rng.choice(len(indices), max_per_task, replace=False)
            sampled.extend(indices[i] for i in chosen)
        else:
            sampled.extend(indices)

    subset_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"max_per_task": max_per_task, "index": sampled},
              open(subset_path, "w"))
    log.info(f"Saved {len(sampled)} subset indices → {subset_path}")
    new_ds        = copy.copy(dataset)
    new_ds._index = sampled
    return new_ds


# ── evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(policy: SmolVLAPolicy, loader: DataLoader,
             device: str, action_dim: int,
             chunk_size: int, success_thresh: float) -> dict:
    """
    Single pass computing every metric — standard and non-standard.

    Standard
    --------
    flow_loss, mean_mae, mean_mse, per_dim_mae, per_dim_mse, chunk_mae

    Non-standard
    ------------
    median_mae, p90_mae, success_rate, avg_steps_before_fail,
    directional_accuracy, gripper_accuracy, smoothness_ratio,
    temporal_drift_slope, action_magnitude_bias
    """

    # ── standard accumulators ─────────────────────────────────────────────
    flow_total, flow_n = 0.0, 0
    sum_abs     = torch.zeros(action_dim, device=device)
    sum_sq      = torch.zeros(action_dim, device=device)
    total_steps = torch.tensor(0.0, device=device)

    third  = chunk_size // 3
    thirds = [(0, third), (third, 2 * third), (2 * third, chunk_size)]
    chunk_abs = [torch.zeros(action_dim, device=device) for _ in thirds]
    chunk_cnt = [torch.tensor(0.0,       device=device) for _ in thirds]

    # ── non-standard accumulators ─────────────────────────────────────────

    # per-episode MAE list → median, P90, success_rate
    per_episode_maes: list[torch.Tensor] = []

    # per-timestep MAE (chunk_size,) → temporal drift slope
    per_step_abs_sum = torch.zeros(chunk_size, device=device)
    per_step_cnt_sum = torch.zeros(chunk_size, device=device)

    # directional accuracy: sign(pred) == sign(gt)
    dir_correct = torch.tensor(0.0, device=device)
    dir_total   = torch.tensor(0.0, device=device)

    # gripper accuracy: last action dimension only
    grip_correct = torch.tensor(0.0, device=device)
    grip_total   = torch.tensor(0.0, device=device)

    # smoothness: mean |a_{t+1} - a_t| for pred vs gt
    pred_delta_sum = torch.tensor(0.0, device=device)
    gt_delta_sum   = torch.tensor(0.0, device=device)
    delta_n        = torch.tensor(0.0, device=device)

    # action magnitude bias: mean(|pred|) / mean(|gt|) per dim
    pred_mag_sum = torch.zeros(action_dim, device=device)
    gt_mag_sum   = torch.zeros(action_dim, device=device)

    # steps before first failure
    steps_before_fail: list[float] = []

    sample_count = 0

    for batch in tqdm(loader, desc="evaluating", leave=True):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # ── flow loss (fast single forward, no denoising) ────────────────
        loss, _ = policy(batch)
        flow_total += loss.item()
        flow_n     += 1

        # ── full denoising → predicted action chunk ──────────────────────
        images, img_masks = policy.prepare_images(batch)
        state             = policy.prepare_state(batch)
        lang_tokens       = batch["observation.language_tokens"]
        lang_masks        = batch["observation.language_attention_mask"]

        pred = policy.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state
        )                                   # (B, chunk_size, max_action_dim)
        pred = pred[:, :, :action_dim]      # (B, chunk_size, action_dim)

        gt     = batch["action"]            # (B, chunk_size, action_dim)
        is_pad = batch["action_is_pad"]     # (B, chunk_size)
        mask   = (~is_pad).float()          # (B, chunk_size)

        B     = pred.shape[0]
        err   = pred - gt
        abs_e = err.abs()
        sq_e  = err ** 2

        masked_abs = abs_e * mask.unsqueeze(-1)
        masked_sq  = sq_e  * mask.unsqueeze(-1)

        # ── standard: global MAE / MSE ───────────────────────────────────
        sum_abs     += masked_abs.sum(dim=(0, 1))
        sum_sq      += masked_sq .sum(dim=(0, 1))
        total_steps += mask.sum()

        # ── standard: chunk thirds ───────────────────────────────────────
        for i, (s, e) in enumerate(thirds):
            seg_mask = mask[:, s:e]
            seg_abs  = abs_e[:, s:e, :] * seg_mask.unsqueeze(-1)
            chunk_abs[i] += seg_abs.sum(dim=(0, 1))
            chunk_cnt[i] += seg_mask.sum()

        # ── non-standard 1: per-episode MAE (for median, P90, success) ───
        ep_denom = mask.sum(dim=1).clamp(min=1)                    # (B,)
        ep_mae   = masked_abs.sum(dim=(1, 2)) / ep_denom / action_dim  # (B,)
        per_episode_maes.append(ep_mae.cpu())

        # ── non-standard 2: per-step MAE (temporal drift) ────────────────
        step_mae = abs_e.mean(dim=2) * mask                        # (B, chunk)
        per_step_abs_sum += step_mae.sum(dim=0)
        per_step_cnt_sum += mask.sum(dim=0)

        # ── non-standard 3: directional accuracy ─────────────────────────
        # count only non-padded steps where gt != 0 (sign is meaningful)
        gt_nonzero = (gt.abs() > 1e-6).float()
        valid_3d   = mask.unsqueeze(-1) * gt_nonzero
        dir_match  = (torch.sign(pred) == torch.sign(gt)).float() * valid_3d
        dir_correct += dir_match.sum()
        dir_total   += valid_3d.sum()

        # ── non-standard 4: gripper accuracy ─────────────────────────────
        grip_gt   = gt  [:, :, -1]
        grip_pred = pred[:, :, -1]
        grip_nz   = (grip_gt.abs() > 1e-6).float()
        grip_valid = mask * grip_nz
        grip_match = (torch.sign(grip_pred) == torch.sign(grip_gt)).float()
        grip_correct += (grip_match * grip_valid).sum()
        grip_total   += grip_valid.sum()

        # ── non-standard 5: action smoothness ────────────────────────────
        if chunk_size > 1:
            # both consecutive steps must be non-padded
            consec_mask    = mask[:, :-1] * mask[:, 1:]           # (B, chunk-1)
            pred_step_diff = (pred[:, 1:, :] - pred[:, :-1, :]).abs()
            gt_step_diff   = (gt  [:, 1:, :] - gt  [:, :-1, :]).abs()
            cm3             = consec_mask.unsqueeze(-1)
            pred_delta_sum += (pred_step_diff * cm3).sum()
            gt_delta_sum   += (gt_step_diff   * cm3).sum()
            delta_n        += (consec_mask * action_dim).sum()

        # ── non-standard 6: action magnitude bias ────────────────────────
        pred_mag_sum += masked_abs.sum(dim=(0, 1))
        gt_mag_sum   += (gt.abs() * mask.unsqueeze(-1)).sum(dim=(0, 1))

        # ── non-standard 7: steps before first failure ───────────────────
        # For each episode sample: first chunk step where step-level error
        # exceeds success_thresh, or total valid steps if never fails.
        step_mae_cpu = step_mae.cpu()
        mask_cpu     = mask.cpu()
        for b_i in range(B):
            valid = mask_cpu[b_i].bool()
            if not valid.any():
                continue
            errs      = step_mae_cpu[b_i]
            fail_mask = (errs > success_thresh) & valid
            if fail_mask.any():
                first_fail = int(fail_mask.nonzero(as_tuple=True)[0][0].item())
                steps_before_fail.append(float(first_fail))
            else:
                steps_before_fail.append(float(valid.sum().item()))

        sample_count += B

    if total_steps == 0:
        return {}

    # ── aggregate: standard ───────────────────────────────────────────────
    per_dim_mae = (sum_abs / total_steps).cpu().tolist()
    per_dim_mse = (sum_sq  / total_steps).cpu().tolist()
    mean_mae    = float(np.mean(per_dim_mae))
    mean_mse    = float(np.mean(per_dim_mse))
    flow_loss   = flow_total / max(flow_n, 1)

    chunk_mae_vals = []
    for i in range(3):
        cnt  = chunk_cnt[i].item()
        vals = (chunk_abs[i] / cnt).cpu().tolist() if cnt > 0 else [0.0] * action_dim
        chunk_mae_vals.append(float(np.mean(vals)))

    # ── aggregate: non-standard ───────────────────────────────────────────

    all_ep_maes  = torch.cat(per_episode_maes)
    success_rate = float((all_ep_maes < success_thresh).float().mean() * 100)
    median_mae   = float(all_ep_maes.median())
    p90_mae      = float(all_ep_maes.quantile(0.9))

    # temporal drift: linear regression slope of per-step MAE over timestep
    valid_mask   = per_step_cnt_sum > 0
    step_indices = valid_mask.nonzero(as_tuple=True)[0].float().cpu().numpy()
    step_maes    = (per_step_abs_sum[valid_mask]
                    / per_step_cnt_sum[valid_mask]).cpu().numpy()
    drift_slope  = (float(np.polyfit(step_indices, step_maes, 1)[0])
                    if len(step_indices) >= 2 else _NAN)
    per_step_mae_list = (per_step_abs_sum
                         / per_step_cnt_sum.clamp(min=1)).cpu().tolist()

    # directional accuracy
    dir_acc  = (float(dir_correct  / dir_total .clamp(min=1)) * 100
                if dir_total  > 0 else _NAN)
    grip_acc = (float(grip_correct / grip_total.clamp(min=1)) * 100
                if grip_total > 0 else _NAN)

    # smoothness ratio
    if delta_n > 0:
        pred_smooth      = float(pred_delta_sum / delta_n)
        gt_smooth        = float(gt_delta_sum   / delta_n)
        smoothness_ratio = pred_smooth / gt_smooth if gt_smooth > 1e-8 else _NAN
    else:
        pred_smooth = gt_smooth = smoothness_ratio = _NAN

    # action magnitude bias per dim
    mag_bias = (pred_mag_sum / gt_mag_sum.clamp(min=1e-8)).cpu().tolist()

    # steps before failure
    avg_steps_before_fail = (float(np.mean(steps_before_fail))
                             if steps_before_fail else _NAN)

    return {
        # standard
        "flow_loss":   flow_loss,
        "mean_mae":    mean_mae,
        "mean_mse":    mean_mse,
        "per_dim_mae": per_dim_mae,
        "per_dim_mse": per_dim_mse,
        "chunk_mae": {
            "early": chunk_mae_vals[0],
            "mid":   chunk_mae_vals[1],
            "late":  chunk_mae_vals[2],
        },
        # non-standard
        "median_mae":             median_mae,
        "p90_mae":                p90_mae,
        "success_rate":           success_rate,
        "avg_steps_before_fail":  avg_steps_before_fail,
        "directional_accuracy":   dir_acc,
        "gripper_accuracy":       grip_acc,
        "pred_smoothness":        pred_smooth,
        "gt_smoothness":          gt_smooth,
        "smoothness_ratio":       smoothness_ratio,
        "temporal_drift_slope":   drift_slope,
        "action_magnitude_bias":  mag_bias,
        "per_step_mae":           per_step_mae_list,
        "n_samples":              sample_count,
    }


# ── report printing ───────────────────────────────────────────────────────────

def _nan_safe(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return _NAN


def _fmt_delta(b, f, lower_is_better=True):
    b, f = _nan_safe(b), _nan_safe(f)
    if b != b or f != f:
        return _NAN, _NAN, "?"
    d     = f - b
    pct   = (d / b * 100) if abs(b) > 1e-12 else _NAN
    arrow = ("↓" if d < 0 else "↑") if lower_is_better else ("↑" if d > 0 else "↓")
    return d, pct, arrow


def print_report(base: dict, fine: dict,
                 base_label: str, fine_label: str,
                 action_dim: int, success_thresh: float):

    W            = 80
    joint_labels = [f"j{i+1}" for i in range(action_dim - 1)] + ["grip"]

    def section(title):
        print(f"\n  {'─' * (W - 4)}")
        print(f"  {title}")

    def hdr():
        print(f"\n  {'Metric':<32}  {'Baseline':>11}  {'Finetuned':>11}  "
              f"{'Delta':>9}  {'Change':>9}")
        print(f"  {'─'*32}  {'─'*11}  {'─'*11}  {'─'*9}  {'─'*9}")

    def row(name, bv, fv, fmt=".4f", lower_is_better=True):
        bv, fv    = _nan_safe(bv), _nan_safe(fv)
        d, pct, arrow = _fmt_delta(bv, fv, lower_is_better)
        b_s = f"{bv:{fmt}}" if bv == bv else "   n/a"
        f_s = f"{fv:{fmt}}" if fv == fv else "   n/a"
        d_s = f"{d:>+9.4f}" if d == d else "     n/a"
        p_s = f"{arrow}{abs(pct):>5.1f}%" if pct == pct else "    n/a"
        print(f"  {name:<32}  {b_s:>11}  {f_s:>11}  {d_s}  {p_s}")

    # ── header ────────────────────────────────────────────────────────────
    print("\n" + "=" * W)
    print(f"  SmolVLA Model Comparison  —  SO-100 Pick-and-Place")
    print("=" * W)
    print(f"  Baseline  : {base_label}")
    print(f"  Finetuned : {fine_label}")
    print(f"  Samples   : {base.get('n_samples','?')}  |  "
          f"action_dim: {action_dim}  |  "
          f"success_thresh: {success_thresh}")

    # ── 1. Training loss ──────────────────────────────────────────────────
    section("1. Training Loss")
    hdr()
    row("flow_loss", base["flow_loss"], fine["flow_loss"])

    # ── 2. Standard action error ──────────────────────────────────────────
    section("2. Standard Action Error  (normalized units, lower = better)")
    hdr()
    row("mean_mae", base["mean_mae"], fine["mean_mae"])
    row("mean_mse", base["mean_mse"], fine["mean_mse"])

    # ── 3. Robust error statistics ────────────────────────────────────────
    section("3. Robust Error Distribution  (per-episode)")
    print(f"  {'':32}  {'':11}  {'':11}  "
          f"  note")
    print(f"  {'─'*32}  {'─'*11}  {'─'*11}  {'─'*22}")
    def row_note(name, bv, fv, note, fmt=".4f", lower_is_better=True):
        bv, fv        = _nan_safe(bv), _nan_safe(fv)
        d, pct, arrow = _fmt_delta(bv, fv, lower_is_better)
        b_s = f"{bv:{fmt}}" if bv == bv else "   n/a"
        f_s = f"{fv:{fmt}}" if fv == fv else "   n/a"
        p_s = f"{arrow}{abs(pct):>5.1f}%" if pct == pct else "    n/a"
        print(f"  {name:<32}  {b_s:>11}  {f_s:>11}  {p_s:>9}  {note}")

    row_note("median_mae",
             base.get("median_mae", _NAN), fine.get("median_mae", _NAN),
             "robust central tendency (less skewed by outliers)")
    row_note("p90_mae  (worst 10% of episodes)",
             base.get("p90_mae", _NAN), fine.get("p90_mae", _NAN),
             "tail performance / worst-case behavior")

    # ── 4. Task success proxies ───────────────────────────────────────────
    section("4. Task Success Proxies  (offline substitutes for live robot eval)")
    print(f"  {'':32}  {'':11}  {'':11}  "
          f"  note")
    print(f"  {'─'*32}  {'─'*11}  {'─'*11}  {'─'*22}")
    row_note(f"success_rate (%)  [MAE<{success_thresh}]",
             base.get("success_rate", _NAN), fine.get("success_rate", _NAN),
             "% episodes below error thresh  (higher=better)",
             fmt=".2f", lower_is_better=False)
    row_note("avg_steps_before_fail",
             base.get("avg_steps_before_fail", _NAN),
             fine.get("avg_steps_before_fail", _NAN),
             "steps until step-error>thresh  (higher=lasts longer)",
             fmt=".1f", lower_is_better=False)

    # ── 5. Behavioral quality (the non-standard metrics) ──────────────────
    section("5. Behavioral Quality  ← the non-standard metrics")
    print(f"  {'':32}  {'':11}  {'':11}  "
          f"  note")
    print(f"  {'─'*32}  {'─'*11}  {'─'*11}  {'─'*30}")
    row_note("directional_accuracy (%)",
             base.get("directional_accuracy", _NAN),
             fine.get("directional_accuracy", _NAN),
             "sign(pred)==sign(gt): right direction?  (higher=better)",
             fmt=".2f", lower_is_better=False)
    row_note("gripper_accuracy (%)",
             base.get("gripper_accuracy", _NAN),
             fine.get("gripper_accuracy", _NAN),
             "binary open/close accuracy  (higher=better)",
             fmt=".2f", lower_is_better=False)
    row_note("smoothness_ratio",
             base.get("smoothness_ratio", _NAN),
             fine.get("smoothness_ratio", _NAN),
             "pred jerk / gt jerk  (1.0=matches demo, >1 jerky)",
             fmt=".4f", lower_is_better=False)
    row_note("temporal_drift_slope",
             base.get("temporal_drift_slope", _NAN),
             fine.get("temporal_drift_slope", _NAN),
             "slope of per-step MAE over chunk  (lower=more consistent)",
             fmt=".6f", lower_is_better=True)

    # ── 6. Action magnitude bias ──────────────────────────────────────────
    section("6. Action Magnitude Bias  mean(|pred|) / mean(|gt|) per joint")
    print(f"  (1.0 = matches demo magnitude; <1 underestimates; >1 overestimates)")
    print(f"  {'Joint':<10}  {'Baseline':>11}  {'Finetuned':>11}  {'Delta':>9}  interpretation")
    print(f"  {'─'*10}  {'─'*11}  {'─'*11}  {'─'*9}  {'─'*26}")
    base_bias = base.get("action_magnitude_bias", [_NAN] * action_dim)
    fine_bias = fine.get("action_magnitude_bias", [_NAN] * action_dim)
    for lbl, bv, fv in zip(joint_labels, base_bias, fine_bias):
        bv, fv = _nan_safe(bv), _nan_safe(fv)
        d      = fv - bv if (bv == bv and fv == fv) else _NAN
        interp = ""
        if fv == fv:
            if   fv < 0.70: interp = "<< severe underestimate (lazy model)"
            elif fv < 0.90: interp = "<  slight underestimate"
            elif fv > 1.30: interp = ">> severe overestimate (over-shoots)"
            elif fv > 1.10: interp = ">  slight overestimate"
            else:            interp = "~  matches demo magnitude"
        b_s = f"{bv:.4f}" if bv == bv else "   n/a"
        f_s = f"{fv:.4f}" if fv == fv else "   n/a"
        d_s = f"{d:>+9.4f}" if d == d else "     n/a"
        print(f"  {lbl:<10}  {b_s:>11}  {f_s:>11}  {d_s}  {interp}")

    # ── 7. Per-joint MAE ──────────────────────────────────────────────────
    section("7. Per-Joint MAE  (normalized units, lower = better)")
    hdr()
    for lbl, bv, fv in zip(joint_labels,
                            base.get("per_dim_mae", [_NAN] * action_dim),
                            fine.get("per_dim_mae", [_NAN] * action_dim)):
        row(lbl, bv, fv)

    # ── 8. Action-chunk temporal breakdown ───────────────────────────────
    section("8. Action-Chunk MAE by Planning Horizon  (early / mid / late)")
    hdr()
    row("early  (first 1/3 of chunk)",
        base.get("chunk_mae", {}).get("early", _NAN),
        fine.get("chunk_mae", {}).get("early", _NAN))
    row("mid    (middle 1/3)",
        base.get("chunk_mae", {}).get("mid", _NAN),
        fine.get("chunk_mae", {}).get("mid", _NAN))
    row("late   (last 1/3 of chunk)",
        base.get("chunk_mae", {}).get("late", _NAN),
        fine.get("chunk_mae", {}).get("late", _NAN))

    # ── verdict ───────────────────────────────────────────────────────────
    checks = [
        ("flow_loss",          base.get("flow_loss",            _NAN), fine.get("flow_loss",            _NAN), True),
        ("mean_mae",           base.get("mean_mae",             _NAN), fine.get("mean_mae",             _NAN), True),
        ("success_rate",       base.get("success_rate",         _NAN), fine.get("success_rate",         _NAN), False),
        ("directional_acc",    base.get("directional_accuracy", _NAN), fine.get("directional_accuracy", _NAN), False),
        ("gripper_acc",        base.get("gripper_accuracy",     _NAN), fine.get("gripper_accuracy",     _NAN), False),
        ("steps_before_fail",  base.get("avg_steps_before_fail",_NAN), fine.get("avg_steps_before_fail",_NAN), False),
        ("temporal_drift",     base.get("temporal_drift_slope", _NAN), fine.get("temporal_drift_slope", _NAN), True),
    ]
    n_improved, n_valid = 0, 0
    for _, bv, fv, lib in checks:
        bv, fv = _nan_safe(bv), _nan_safe(fv)
        if bv != bv or fv != fv:
            continue
        n_valid += 1
        if (lib and fv < bv) or (not lib and fv > bv):
            n_improved += 1

    print("\n" + "=" * W)
    if   n_valid == 0:                          verdict = "No metrics available."
    elif n_improved == n_valid:                 verdict = f"Fine-tuning improved ALL {n_valid} tracked metrics."
    elif n_improved >= max(1, n_valid * 3 // 4): verdict = f"Fine-tuning improved {n_improved}/{n_valid} tracked metrics."
    elif n_improved >= max(1, n_valid // 2):    verdict = f"Fine-tuning improved {n_improved}/{n_valid} metrics — mixed results."
    else:                                       verdict = (f"Only {n_improved}/{n_valid} metrics improved. "
                                                           "Consider more steps or a lower LR.")
    print(f"  Verdict: {verdict}")
    for name, bv, fv, lib in checks:
        bv, fv = _nan_safe(bv), _nan_safe(fv)
        if bv != bv or fv != fv:
            continue
        _, pct, arrow = _fmt_delta(bv, fv, lib)
        unit  = " pp" if "rate" in name or "acc" in name else "%"
        p_str = f"{arrow}{abs(pct):.1f}{unit}" if pct == pct else "n/a"
        print(f"    {name:<24}: {p_str}")
    print("=" * W + "\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = args.device

    # ── Option B: pre-computed JSONs (no inference) ───────────────────────
    if args.baseline_json and args.finetuned_json:
        base = json.load(open(args.baseline_json))
        fine = json.load(open(args.finetuned_json))
        action_dim = base.get("action_dim", len(base.get("per_dim_mae", [])))

        _extra_keys = [
            "mean_mse", "median_mae", "p90_mae", "success_rate",
            "avg_steps_before_fail", "directional_accuracy",
            "gripper_accuracy", "smoothness_ratio",
            "temporal_drift_slope",
        ]
        for d in (base, fine):
            for k in _extra_keys:
                d.setdefault(k, _NAN)
            d.setdefault("per_dim_mse", [_NAN] * action_dim)
            d.setdefault("action_magnitude_bias", [_NAN] * action_dim)
            d.setdefault("chunk_mae", {"early": _NAN, "mid": _NAN, "late": _NAN})

        log.warning(
            "Non-standard metrics show n/a (pre-computed JSONs don't contain them). "
            "Use --baseline / --finetuned to compute all metrics."
        )
        print_report(base, fine,
                     base["checkpoint"], fine["checkpoint"],
                     action_dim, args.success_thresh)

        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            json.dump({"baseline": base, "finetuned": fine},
                      open(out, "w"), indent=2)
            log.info(f"Saved → {out}")
        return

    # ── Option A: run inference ───────────────────────────────────────────
    if not (args.baseline and args.finetuned):
        raise ValueError(
            "Provide either (--baseline + --finetuned) to run inference, "
            "or (--baseline_json + --finetuned_json) for pre-computed results."
        )

    meta_path = Path(args.data) / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {args.data}.")
    with open(meta_path) as f:
        meta = json.load(f)
    action_dim = meta["action_dim"]
    state_dim  = meta["state_dim"]
    log.info(f"SO-100: action_dim={action_dim}, state_dim={state_dim}")

    norm_path  = Path(args.data) / "norm_stats.json"
    norm_stats = None
    if norm_path.exists():
        norm_stats = json.load(open(norm_path))
        log.info("Normalization stats loaded.")
    else:
        log.warning("No norm_stats.json — running without normalization.")

    dataset = build_dataset(args, norm_stats)
    log.info(f"Split '{args.split}': {len(dataset)} samples")

    if args.max_per_task is not None:
        dataset = subsample(dataset, args.max_per_task,
                            args.subset_file, args.seed)
        log.info(f"Subsampled to {len(dataset)} samples "
                 f"(≤{args.max_per_task} per task)")

    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    results = {}
    for label, ckpt in [("baseline",  args.baseline),
                         ("finetuned", args.finetuned)]:
        log.info(f"\n{'='*60}")
        log.info(f"Evaluating {label}: {ckpt}")
        policy  = load_policy(ckpt, action_dim=action_dim,
                              state_dim=state_dim, device=device,
                              chunk_size=args.chunk)
        metrics = evaluate(policy, loader, device,
                           action_dim=action_dim,
                           chunk_size=args.chunk,
                           success_thresh=args.success_thresh)
        metrics["checkpoint"] = ckpt
        metrics["split"]      = args.split
        results[label]        = metrics

        log.info(f"  flow_loss             = {metrics['flow_loss']:.6f}")
        log.info(f"  mean_mae              = {metrics['mean_mae']:.6f}")
        log.info(f"  median_mae            = {metrics['median_mae']:.6f}")
        log.info(f"  p90_mae               = {metrics['p90_mae']:.6f}")
        log.info(f"  success_rate          = {metrics['success_rate']:.2f}%")
        log.info(f"  directional_accuracy  = {metrics['directional_accuracy']:.2f}%")
        log.info(f"  gripper_accuracy      = {metrics['gripper_accuracy']:.2f}%")
        log.info(f"  smoothness_ratio      = {metrics['smoothness_ratio']:.4f}")
        log.info(f"  temporal_drift_slope  = {metrics['temporal_drift_slope']:.6f}")
        log.info(f"  avg_steps_before_fail = {metrics['avg_steps_before_fail']:.1f}")

        del policy
        if device == "cuda":
            torch.cuda.empty_cache()

    print_report(results["baseline"], results["finetuned"],
                 args.baseline, args.finetuned,
                 action_dim, args.success_thresh)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(out, "w"), indent=2)
        log.info(f"Full comparison saved → {out}")


if __name__ == "__main__":
    main()
