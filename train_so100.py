"""
train_so100.py
==============
Fine-tune SmolVLAPolicy on the lerobot/svla_so100_pickplace dataset.

Reads data/so100_pickplace/metadata.json to auto-configure action_dim
and state_dim (typically 6 for the SO-100 robot arm).

Usage:
    # Recommended — start from pretrained SmolVLA, train expert only
    python train_so100.py \
        --data      data/so100_pickplace \
        --output    checkpoints/so100_run \
        --steps     5000 \
        --batch     4 \
        --lr        1e-4 \
        --bf16

    # Smaller run for quick testing
    python train_so100.py \
        --data   data/so100_pickplace \
        --output checkpoints/so100_test \
        --steps  500 \
        --batch  2

    # Full fine-tune (slower, more GPU RAM)
    python train_so100.py \
        --data   data/so100_pickplace \
        --output checkpoints/so100_full \
        --steps  10000 \
        --no_train_expert_only
"""

import argparse
import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config import FeatureType, PolicyFeature, SmolVLAConfig
from dataset_so100 import SO100Dataset, make_splits
from model import SmolVLAPolicy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_so100")


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune SmolVLA on SO-100 pick-and-place"
    )
    p.add_argument("--data",             default="data/so100_pickplace")
    p.add_argument("--output",           default="checkpoints/so100_run")
    p.add_argument("--from_pretrained",  default="lerobot/smolvla_base",
                   help="HF repo or local checkpoint to start from. "
                        "Set to '' to train from random init.")
    p.add_argument("--steps",            type=int,   default=5000)
    p.add_argument("--batch",            type=int,   default=4)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--weight_decay",     type=float, default=1e-10)
    p.add_argument("--grad_clip",        type=float, default=10.0)
    p.add_argument("--warmup_steps",     type=int,   default=200)
    p.add_argument("--chunk",            type=int,   default=50)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--num_workers",      type=int,   default=2)
    p.add_argument("--log_every",        type=int,   default=50)
    p.add_argument("--save_every",       type=int,   default=500)
    p.add_argument("--val_every",        type=int,   default=500)
    p.add_argument("--val_fraction",     type=float, default=0.1)
    p.add_argument("--no_train_expert_only", action="store_true",
                   help="Un-freeze full model (slower but potentially better)")
    p.add_argument("--bf16", action="store_true",
                   help="Use bfloat16 mixed precision (recommended on A100/H100)")

    # ── Magnitude-collapse fixes ────────────────────────────────────────
    p.add_argument("--num_steps", type=int, default=10,
                   help="Flow-matching denoising steps at inference. "
                        "Increase to 50 to fix magnitude underestimation on "
                        "high-scale actions. (default: 10)")
    p.add_argument("--mag_loss_weight", type=float, default=0.0,
                   help="Weight for velocity-magnitude alignment loss. "
                        "0=off (default). Try 0.1–0.5 if action magnitude "
                        "bias < 0.5 after retraining with fixed norm_stats.")
    p.add_argument("--action_dim_weights", type=str, default=None,
                   help="Comma-separated per-dim loss weights, e.g. '1,3,3,3,3,2'. "
                        "Upweights underestimated joints. None = uniform.")
    return p.parse_args()


# ── model setup ────────────────────────────────────────────────────────────

def build_model(args, action_dim: int, state_dim: int):
    """Load SmolVLAPolicy and patch with SO-100 feature shapes."""
    train_expert_only = not args.no_train_expert_only

    input_features = {
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
    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))
    }

    # Parse per-dim loss weights
    dim_weights = None
    if args.action_dim_weights:
        dim_weights = [float(x) for x in args.action_dim_weights.split(",")]
        log.info(f"Per-dim loss weights: {dim_weights}")

    if args.from_pretrained:
        log.info(f"Loading pretrained weights from {args.from_pretrained}")
        policy = SmolVLAPolicy.from_pretrained(args.from_pretrained)
        # Patch feature shapes to match SO-100
        policy.config.input_features  = input_features
        policy.config.output_features = output_features
    else:
        log.info("Building SmolVLA from scratch (no pretrained weights)")
        config = SmolVLAConfig(
            input_features=input_features,
            output_features=output_features,
            vlm_model_name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
            load_vlm_weights=True,
            train_expert_only=train_expert_only,
            freeze_vision_encoder=True,
            chunk_size=args.chunk,
            n_action_steps=args.chunk,
            num_steps=10,
        )
        policy = SmolVLAPolicy(config)

    # Sync chunk size — MUST be set before any forward pass.
    # embed_suffix uses self.config.chunk_size for att_masks length,
    # which must match the actual action tensor length from the DataLoader.
    # If these differ (e.g. loaded checkpoint had chunk=50 but --chunk 20),
    # make_att_2d_masks throws a size-mismatch RuntimeError.
    policy.config.chunk_size     = args.chunk
    policy.config.n_action_steps = args.chunk
    log.info(f"Chunk size set to {args.chunk}")

    # Apply magnitude-collapse fixes
    policy.config.num_steps        = args.num_steps
    policy.config.mag_loss_weight  = args.mag_loss_weight
    policy.config.action_dim_weights = dim_weights
    if args.num_steps != 10:
        log.info(f"Denoising steps set to {args.num_steps}")
    if args.mag_loss_weight > 0:
        log.info(f"Magnitude alignment loss enabled (weight={args.mag_loss_weight})")

    # Freeze / unfreeze
    policy.config.train_expert_only = train_expert_only
    for name, param in policy.named_parameters():
        if train_expert_only:
            trainable = any(
                name.startswith(prefix)
                for prefix in (
                    "model.lm_expert",
                    "model.vlm_with_expert.lm_expert",
                    "model.state_proj",
                    "model.action_in_proj",
                    "model.action_out_proj",
                    "model.action_time_mlp",
                )
            )
            param.requires_grad = trainable
        else:
            param.requires_grad = True

    n_total     = sum(p.numel() for p in policy.parameters())
    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    log.info(
        f"Parameters: {n_total/1e6:.1f}M total, "
        f"{n_trainable/1e6:.1f}M trainable "
        f"({'expert-only' if train_expert_only else 'full model'})"
    )

    return policy


# ── LR schedule ────────────────────────────────────────────────────────────

class WarmupCosineSchedule:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 min_lr_ratio: float = 0.1):
        self.optimizer    = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs     = [pg["lr"] for pg in optimizer.param_groups]
        self._step        = 0

    def step(self):
        self._step += 1
        s = self._step
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if s <= self.warmup_steps:
                lr = base_lr * s / max(1, self.warmup_steps)
            else:
                progress = (s - self.warmup_steps) / max(
                    1, self.total_steps - self.warmup_steps
                )
                lr = base_lr * (
                    self.min_lr_ratio
                    + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
                )
            pg["lr"] = lr


# ── eval ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(policy: nn.Module, val_loader: DataLoader, device: str,
             scaler=None) -> float:
    policy.eval()
    total_loss = 0.0
    n = 0
    for batch in val_loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, _ = policy(batch)
        else:
            loss, _ = policy(batch)
        total_loss += loss.item()
        n += 1
    policy.train()
    return total_loss / max(n, 1)


# ── training loop ──────────────────────────────────────────────────────────

def train():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device  = args.device
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Read SO-100 metadata ───────────────────────────────────────────
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
        log.info(f"Loaded normalization stats from {norm_path}")
    else:
        log.warning(
            f"No norm_stats.json at {norm_path}. "
            "Run:  python normalize.py --data " + args.data
        )

    # ── Datasets & loaders ─────────────────────────────────────────────
    log.info("Building datasets ...")
    train_ds, val_ds = make_splits(
        args.data,
        chunk_size=args.chunk,
        norm_stats=norm_stats,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    log.info(f"  train: {len(train_ds)} samples | val: {len(val_ds)} samples")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    # ── Model ──────────────────────────────────────────────────────────
    log.info("Building model ...")
    policy = build_model(args, action_dim=action_dim, state_dim=state_dim)
    policy = policy.to(device)
    policy.train()

    # ── BF16 mixed precision ───────────────────────────────────────────
    use_bf16 = args.bf16 and device == "cuda" and torch.cuda.is_bf16_supported()
    if use_bf16:
        log.info("Using bfloat16 mixed precision")
    elif args.bf16:
        log.warning("--bf16 requested but not supported — falling back to fp32")
        use_bf16 = False

    # ── Optimizer & scheduler ──────────────────────────────────────────
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=args.steps,
    )

    # ── Training loop ──────────────────────────────────────────────────
    log.info(f"Starting training for {args.steps} steps on {device}")
    log.info(f"Checkpoints → {out_dir}")

    step       = 0
    loss_accum = 0.0
    t_start    = time.time()

    def infinite_loader(loader):
        while True:
            for batch in loader:
                yield batch

    data_iter = infinite_loader(train_loader)

    while step < args.steps:
        for batch in data_iter:
            if step >= args.steps:
                break

            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            optimizer.zero_grad()

            if use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss, _ = policy(batch)
            else:
                loss, _ = policy(batch)

            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()
            scheduler.step()

            loss_accum += loss.item()
            step += 1

            # ── Logging ───────────────────────────────────────────────
            if step % args.log_every == 0:
                avg_loss = loss_accum / args.log_every
                lr_now   = optimizer.param_groups[0]["lr"]
                elapsed  = time.time() - t_start
                sps      = step / elapsed
                log.info(
                    f"step {step:6d}/{args.steps} | "
                    f"loss {avg_loss:.4f} | "
                    f"lr {lr_now:.2e} | "
                    f"{sps:.1f} steps/s"
                )
                loss_accum = 0.0

            # ── Validation ────────────────────────────────────────────
            if step % args.val_every == 0:
                val_loss = evaluate(policy, val_loader, device)
                log.info(f"  [val] step {step} | val_loss {val_loss:.4f}")

            # ── Checkpoint ────────────────────────────────────────────
            if step % args.save_every == 0:
                ckpt_dir = out_dir / f"step_{step:07d}"
                policy.save_pretrained(str(ckpt_dir))
                log.info(f"  [ckpt] saved to {ckpt_dir}")

    # ── Final save ─────────────────────────────────────────────────────
    final_dir = out_dir / "final"
    policy.save_pretrained(str(final_dir))
    log.info(f"Training complete. Final checkpoint at {final_dir}")

    # Save training args + SO-100 metadata for reproducibility
    with open(out_dir / "train_args.json", "w") as f:
        json.dump({**vars(args), "action_dim": action_dim, "state_dim": state_dim},
                  f, indent=2)

    print(f"\nDone! Next step — evaluate the finetuned model:")
    print(f"  python evaluate_so100.py \\")
    print(f"      --checkpoint {final_dir} \\")
    print(f"      --data {args.data} \\")
    print(f"      --out results/so100_finetuned.json")


if __name__ == "__main__":
    train()
