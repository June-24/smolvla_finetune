"""
normalize_robust.py
===================
Recomputes normalization statistics using percentile clipping to prevent
the bimodal pick-and-place action distribution from producing inflated std
values that cause flow-matching magnitude collapse.

Usage:
    python normalize_robust.py --data data/so100_pickplace --out data/so100_pickplace/norm_stats.json

Why this matters
----------------
normalize.py uses raw mean/std.  For pick-and-place, most frames have
near-zero actions (stationary phases) with brief large movements.  This
creates a distribution where:

  - If stationary frames dominate: std is tiny → active frames become 50-150
    std deviations → flow matching can't learn large action magnitudes
  - If active frames dominate: a different bias

This script clips to the [p_low, p_high] percentile range before computing
std, making the normalized data stay within roughly [-3, 3].

It also prints a diagnostic table so you can see the raw action statistics
and verify the normalization is reasonable.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def compute_stats_robust(
    data_dir: str,
    out_path: str | None = None,
    clip_low: float = 1.0,
    clip_high: float = 99.0,
    min_std: float = 1e-3,
):
    data_dir  = Path(data_dir)
    out_path  = Path(out_path) if out_path else data_dir / "norm_stats.json"

    pq_path = data_dir / "data.parquet"
    print(f"Loading {pq_path} ...")

    table  = pq.read_table(str(pq_path), columns=["action", "observation.state"])
    df     = table.to_pandas()

    actions = np.array(df["action"].tolist(),             dtype=np.float32)
    states  = np.array(df["observation.state"].tolist(),  dtype=np.float32)

    def robust_stats(arr: np.ndarray, name: str) -> tuple[list, list]:
        """Clip to [p_low, p_high] percentile range, then compute mean/std."""
        p_lo  = np.percentile(arr, clip_low,  axis=0)
        p_hi  = np.percentile(arr, clip_high, axis=0)
        clipped = np.clip(arr, p_lo, p_hi)

        mean = clipped.mean(axis=0)
        std  = clipped.std(axis=0).clip(min=min_std)

        print(f"\n  {name}  (N={len(arr)}, dims={arr.shape[1]})")
        print(f"  {'dim':<6} {'raw_min':>10} {'raw_max':>10} {'p{:.0f}'.format(clip_low):>10} "
              f"{'p{:.0f}'.format(clip_high):>10} {'mean':>10} {'std':>10} "
              f"  {'norm_range':>18}")
        print(f"  {'─'*6} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10} "
              f"  {'─'*18}")
        for d in range(arr.shape[1]):
            norm_lo = (arr[:, d].min() - mean[d]) / std[d]
            norm_hi = (arr[:, d].max() - mean[d]) / std[d]
            print(f"  {d:<6} {arr[:,d].min():>10.4f} {arr[:,d].max():>10.4f} "
                  f"{p_lo[d]:>10.4f} {p_hi[d]:>10.4f} {mean[d]:>10.4f} {std[d]:>10.4f} "
                  f"  [{norm_lo:+.1f}, {norm_hi:+.1f}]")

        return mean.tolist(), std.tolist()

    print(f"\nPercentile clip range: [{clip_low}%, {clip_high}%]")
    action_mean, action_std = robust_stats(actions, "action")
    state_mean,  state_std  = robust_stats(states,  "state")

    stats = {
        "action": {"mean": action_mean, "std": action_std},
        "state":  {"mean": state_mean,  "std": state_std},
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nNormalization stats saved → {out_path}")
    print("After this, retrain the model and re-evaluate to check if magnitude bias improves.")
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       default="data/so100_pickplace")
    p.add_argument("--out",        default=None)
    p.add_argument("--clip_low",   type=float, default=1.0,
                   help="Lower percentile for clipping before computing stats.")
    p.add_argument("--clip_high",  type=float, default=99.0,
                   help="Upper percentile for clipping.")
    p.add_argument("--min_std",    type=float, default=1e-3,
                   help="Minimum std to prevent division by zero.")
    args = p.parse_args()
    compute_stats_robust(args.data, args.out, args.clip_low, args.clip_high, args.min_std)


if __name__ == "__main__":
    main()
