"""Stub for RTC (Real-Time Control) config — only used as a type in SmolVLAConfig."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RTCConfig:
    """Real-Time Control config — not needed for offline training."""
    enabled: bool = False
    max_queue_size: int = 1
