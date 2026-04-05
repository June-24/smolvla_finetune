"""
compare_so100.py
================
Compare pre- and post-finetune evaluate_so100.py JSON outputs side-by-side.

Usage:
    python compare_so100.py results/so100_pretrained.json results/so100_finetuned.json
"""

import json
import sys
from pathlib import Path


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_so100.py <baseline.json> <finetuned.json>")
        sys.exit(1)

    base = load(sys.argv[1])
    fine = load(sys.argv[2])

    action_dim = base.get("action_dim", len(base.get("per_dim_mae", [])))

    # Build per-dim labels dynamically
    if action_dim > 0:
        joint_labels = [f"j{i+1}" for i in range(action_dim - 1)] + ["grip"]
    else:
        joint_labels = [f"j{i+1}" for i in range(len(base.get("per_dim_mae", [])))]

    print("\n" + "=" * 72)
    print(f"  Baseline   : {base['checkpoint']}")
    print(f"  Finetuned  : {fine['checkpoint']}")
    print(f"  Split      : {base.get('split', '?')}  "
          f"({base.get('n_samples', '?')} samples)")
    print(f"  action_dim : {action_dim}")
    print("=" * 72)

    # ── flow_loss ────────────────────────────────────────────────────
    bl    = base["flow_loss"]
    fl    = fine["flow_loss"]
    delta = fl - bl
    pct   = (delta / bl * 100) if bl != 0 else float("nan")
    arrow = "↓" if delta < 0 else "↑"
    print(f"\n  flow_loss")
    print(f"    baseline  : {bl:.6f}")
    print(f"    finetuned : {fl:.6f}  {arrow} {abs(delta):.6f}  ({pct:+.1f}%)")

    # ── action MAE ───────────────────────────────────────────────────
    if "mean_mae" in base and "mean_mae" in fine:
        bm = base["mean_mae"]
        fm = fine["mean_mae"]
        dm = fm - bm
        pm = (dm / bm * 100) if bm != 0 else float("nan")
        aw = "↓" if dm < 0 else "↑"
        print(f"\n  mean_action_mae")
        print(f"    baseline  : {bm:.6f}")
        print(f"    finetuned : {fm:.6f}  {aw} {abs(dm):.6f}  ({pm:+.1f}%)")

        base_dims = base["per_dim_mae"]
        fine_dims = fine["per_dim_mae"]
        labels    = joint_labels[:len(base_dims)]

        print(f"\n  per_dim_mae  {'baseline':>10}  {'finetuned':>10}  {'delta':>10}")
        print(f"  {'-'*48}")
        for lbl, bv, fv in zip(labels, base_dims, fine_dims):
            d  = fv - bv
            aw = "↓" if d < 0 else "↑"
            print(f"    {lbl:4s}      {bv:10.4f}  {fv:10.4f}  {aw} {abs(d):.4f}")

    # ── summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Lower flow_loss and lower MAE = better.")

    # Quick verdict
    improved_loss = fine["flow_loss"] < base["flow_loss"]
    improved_mae  = (
        "mean_mae" in fine and "mean_mae" in base
        and fine["mean_mae"] < base["mean_mae"]
    )

    if improved_loss and improved_mae:
        loss_pct = abs((fine["flow_loss"] - base["flow_loss"]) / base["flow_loss"] * 100)
        mae_pct  = abs((fine["mean_mae"]  - base["mean_mae"])  / base["mean_mae"]  * 100)
        print(f"\n  Result: Fine-tuning improved both metrics.")
        print(f"    flow_loss  reduced by {loss_pct:.1f}%")
        print(f"    action_mae reduced by {mae_pct:.1f}%")
    elif improved_loss:
        loss_pct = abs((fine["flow_loss"] - base["flow_loss"]) / base["flow_loss"] * 100)
        print(f"\n  Result: flow_loss improved by {loss_pct:.1f}%.")
    else:
        print(f"\n  Result: Metrics did not improve — consider more steps or lower LR.")
    print()


if __name__ == "__main__":
    main()
