"""
train.py
========
Standalone finetuning of SmolVLAPolicy on LIBERO.

NO lerobot training framework is used — this is a plain PyTorch training loop.

Usage:
    python train.py \
        --data      data/libero_subset \
        --output    checkpoints/run1 \
        --steps     5000 \
        --batch     4 \
        --lr        1e-4 \
        --chunk     50 \
        --device    cuda

Optional flags:
    --from_pretrained lerobot/smolvla_base   # start from pretrained weights (recommended)
    --grad_clip      10.0
    --log_every      50
    --save_every     500
    --val_every      500
    --seed           42
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── make our lerobot stubs importable ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from dataset import LiberoDataset, make_splits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Finetune SmolVLA on LIBERO")
    p.add_argument("--data",             default="data/libero_subset")
    p.add_argument("--output",           default="checkpoints/run1")
    p.add_argument("--from_pretrained",  default="lerobot/smolvla_base",
                   help="HF repo or local dir to start from. "
                        "Set to '' to train from random init.")
    p.add_argument("--steps",            type=int,   default=5000)
    p.add_argument("--batch",            type=int,   default=4)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--weight_decay",     type=float, default=1e-10)
    p.add_argument("--grad_clip",        type=float, default=10.0)
    p.add_argument("--warmup_steps",     type=int,   default=200)
    p.add_argument("--chunk",            type=int,   default=50)
    p.add_argument("--device",           default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--num_workers",      type=int,   default=2)
    p.add_argument("--log_every",        type=int,   default=50)
    p.add_argument("--save_every",       type=int,   default=500)
    p.add_argument("--val_every",        type=int,   default=500)
    p.add_argument("--val_fraction",     type=float, default=0.1)
    p.add_argument("--train_expert_only",action="store_true", default=True,
                   help="Freeze VLM backbone, only train action expert (recommended)")
    return p.parse_args()


# ── model setup ────────────────────────────────────────────────────────────

def build_model(args):
    """Load SmolVLAPolicy with LIBERO-appropriate config."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature

    # LIBERO: 8-dim state, 7-dim action, 2 cameras
    input_features = {
        "observation.state": PolicyFeature(
            type=FeatureType.STATE, shape=(8,)
        ),
        "observation.images.image": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 256, 256)
        ),
        "observation.images.image2": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 256, 256)
        ),
    }
    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))
    }

    if args.from_pretrained:
        log.info(f"Loading pretrained weights from {args.from_pretrained}")
        # Load from hub — this downloads the VLM and expert weights
        policy = SmolVLAPolicy.from_pretrained(
            args.from_pretrained,
            # Override feature specs to match LIBERO dims
        )
        # Patch the config with LIBERO features (shapes are correct since
        # SmolVLA uses max_state_dim=32 / max_action_dim=32 projections)
        policy.config.input_features  = input_features
        policy.config.output_features = output_features
    else:
        log.info("Building SmolVLA from scratch (no pretrained weights)")
        config = SmolVLAConfig(
            input_features=input_features,
            output_features=output_features,
            vlm_model_name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
            load_vlm_weights=True,
            train_expert_only=args.train_expert_only,
            freeze_vision_encoder=True,
            chunk_size=args.chunk,
            n_action_steps=args.chunk,
            num_steps=10,
        )
        policy = SmolVLAPolicy(config)

    # freeze / unfreeze according to flag
    policy.config.train_expert_only = args.train_expert_only
    for name, param in policy.named_parameters():
        if args.train_expert_only:
            # Only train: lm_expert, state_proj, action_*_proj, action_time_mlp_*
            # All params are under policy.model.*, so match with that prefix too
            trainable = any(
                name.startswith(prefix)
                for prefix in (
                    "model.lm_expert", "model.vlm_with_expert.lm_expert",
                    "model.state_proj",
                    "model.action_in_proj", "model.action_out_proj",
                    "model.action_time_mlp",
                )
            )
            param.requires_grad = trainable
        else:
            param.requires_grad = True

    n_total     = sum(p.numel() for p in policy.parameters())
    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    log.info(f"Parameters: {n_total/1e6:.1f}M total, {n_trainable/1e6:.1f}M trainable")

    return policy


# ── learning-rate schedule ──────────────────────────────────────────────────

class WarmupCosineSchedule:
    """Linear warmup then cosine decay."""
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
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
                progress = (s - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                lr = base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))
            pg["lr"] = lr


# ── eval ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(policy: nn.Module, val_loader: DataLoader, device: str) -> float:
    policy.eval()
    total_loss = 0.0
    n = 0
    for batch in val_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        loss, _ = policy(batch)
        total_loss += loss.item()
        n += 1
    policy.train()
    return total_loss / max(n, 1)


# ── training loop ──────────────────────────────────────────────────────────

def train():
    args = parse_args()

    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load norm stats ────────────────────────────────────────────────
    norm_path = Path(args.data) / "norm_stats.json"
    norm_stats = None
    if norm_path.exists():
        with open(norm_path) as f:
            norm_stats = json.load(f)
        log.info(f"Loaded normalization stats from {norm_path}")
    else:
        log.warning(
            f"No norm_stats.json found at {norm_path}. "
            "Run `python normalize.py` first for best results."
        )

    # ── datasets & loaders ────────────────────────────────────────────
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

    # ── model ─────────────────────────────────────────────────────────
    log.info("Building model ...")
    policy = build_model(args)
    policy = policy.to(device)
    policy.train()

    # ── optimizer & scheduler ─────────────────────────────────────────
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

    # ── training loop ─────────────────────────────────────────────────
    log.info(f"Starting training for {args.steps} steps on {device}")
    log.info(f"Checkpoints → {out_dir}")

    step       = 0
    epoch      = 0
    loss_accum = 0.0
    t_start    = time.time()

    # infinite data iterator
    def infinite_loader(loader):
        while True:
            for batch in loader:
                yield batch

    data_iter = infinite_loader(train_loader)

    while step < args.steps:
        epoch += 1
        for batch in data_iter:
            if step >= args.steps:
                break

            # move to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # ── forward ──────────────────────────────────────────────
            optimizer.zero_grad()
            loss, loss_dict = policy(batch)

            # ── backward ─────────────────────────────────────────────
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()
            scheduler.step()

            loss_accum += loss.item()
            step += 1

            # ── logging ──────────────────────────────────────────────
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

            # ── validation ───────────────────────────────────────────
            if step % args.val_every == 0:
                val_loss = evaluate(policy, val_loader, device)
                log.info(f"  [val] step {step} | val_loss {val_loss:.4f}")

            # ── checkpoint ───────────────────────────────────────────
            if step % args.save_every == 0:
                ckpt_dir = out_dir / f"step_{step:07d}"
                policy.save_pretrained(str(ckpt_dir))
                log.info(f"  [ckpt] saved to {ckpt_dir}")

    # ── final save ────────────────────────────────────────────────────
    final_dir = out_dir / "final"
    policy.save_pretrained(str(final_dir))
    log.info(f"Training complete. Final checkpoint at {final_dir}")

    # save args
    with open(out_dir / "train_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    train()
