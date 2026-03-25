"""Stub scheduler configs used by SmolVLAConfig as type hints."""

from dataclasses import dataclass


@dataclass
class CosineDecayWithWarmupSchedulerConfig:
    peak_lr: float = 1e-4
    decay_lr: float = 1e-5
    warmup_steps: int = 1000
    decay_steps: int = 100000
