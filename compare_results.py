"""
compare_results.py
==================
Compare two evaluate.py JSON outputs side-by-side.

Usage:
    python compare_results.py results/pretrained_chunk0to9.json results/finetuned_chunk0to9.json
"""

import json
import sys
from pathlib import Path


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_results.py <baseline.json> <finetuned.json>")
        sys.exit(1)

    base = load(sys.argv[1])
    fine = load(sys.argv[2])

    print("\n" + "=" * 70)
    print(f"  Baseline   : {base['checkpoint']}")
    print(f"  Finetuned  : {fine['checkpoint']}")
    print(f"  Split      : {base.get('split', '?')}  ({base.get('n_samples', '?')} samples)")
    print("=" * 70)

    # flow loss
    bl = base["flow_loss"]
    fl = fine["flow_loss"]
    delta = fl - bl
    pct   = (delta / bl * 100) if bl != 0 else float("nan")
    arrow = "↓" if delta < 0 else "↑"
    print(f"\n  flow_loss")
    print(f"    baseline  : {bl:.6f}")
    print(f"    finetuned : {fl:.6f}  {arrow} {abs(delta):.6f}  ({pct:+.1f}%)")

    # action MAE
    if "mean_mae" in base and "mean_mae" in fine:
        bm = base["mean_mae"]
        fm = fine["mean_mae"]
        dm = fm - bm
        pm = (dm / bm * 100) if bm != 0 else float("nan")
        aw = "↓" if dm < 0 else "↑"
        print(f"\n  mean_action_mae")
        print(f"    baseline  : {bm:.6f}")
        print(f"    finetuned : {fm:.6f}  {aw} {abs(dm):.6f}  ({pm:+.1f}%)")

        labels = ["j1", "j2", "j3", "j4", "j5", "j6", "grip"]
        print(f"\n  per_dim_mae  {'baseline':>10}  {'finetuned':>10}  {'delta':>10}")
        print(f"  {'-'*46}")
        for lbl, bv, fv in zip(labels, base["per_dim_mae"], fine["per_dim_mae"]):
            d  = fv - bv
            aw = "↓" if d < 0 else "↑"
            print(f"    {lbl:4s}      {bv:10.4f}  {fv:10.4f}  {aw} {abs(d):.4f}")

    print("\n" + "=" * 70)
    print("  Lower flow_loss and lower MAE = better.\n")


if __name__ == "__main__":
    main()
