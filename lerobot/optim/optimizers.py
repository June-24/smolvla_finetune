"""Stub optimizer configs used by SmolVLAConfig as type hints."""

from dataclasses import dataclass, field


@dataclass
class AdamWConfig:
    lr: float = 1e-4
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 1e-10
