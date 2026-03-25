"""Minimal type stubs for lerobot.configs.types used by SmolVLAConfig."""

from dataclasses import dataclass
from enum import Enum


class FeatureType(str, Enum):
    STATE = "state"
    VISUAL = "visual"
    ACTION = "action"
    ENV_STATE = "env_state"


class NormalizationMode(str, Enum):
    MEAN_STD = "mean_std"
    MIN_MAX = "min_max"
    IDENTITY = "identity"


@dataclass
class PolicyFeature:
    type: FeatureType
    shape: tuple


# Used only by processor_smolvla.py which we don't import
class PipelineFeatureType(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
