"""Minimal stub for lerobot.configs.policies — provides PreTrainedConfig base class."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Optional

from huggingface_hub import hf_hub_download

from lerobot.configs.types import FeatureType, PolicyFeature


_REGISTRY: dict[str, type] = {}


@dataclass
class PreTrainedConfig:
    """Base config class matching the real lerobot PreTrainedConfig fields."""

    type: str = ""

    # Training / hub fields that live on the base class in real lerobot
    device: Optional[str] = None
    use_amp: bool = False
    push_to_hub: bool = False
    repo_id: Optional[str] = None
    private: bool = False
    tags: Optional[list] = None
    license: Optional[str] = None

    # subclass registry so @PreTrainedConfig.register_subclass("smolvla") works
    _registry: ClassVar[dict[str, type]] = _REGISTRY

    def __post_init__(self):
        """Called after dataclass __init__. Subclasses call super().__post_init__()."""
        pass

    # ------------------------------------------------------------------
    # Convenience properties used by SmolVLAPolicy (and other policies)
    # ------------------------------------------------------------------

    @property
    def image_features(self) -> dict:
        """Subset of input_features whose type is VISUAL."""
        features = getattr(self, "input_features", {}) or {}
        return {k: v for k, v in features.items() if v.type == FeatureType.VISUAL}

    @property
    def action_feature(self):
        """The single ACTION feature from output_features."""
        features = getattr(self, "output_features", {}) or {}
        for v in features.values():
            if v.type == FeatureType.ACTION:
                return v
        return None

    @property
    def state_feature(self):
        """The single STATE feature from input_features."""
        features = getattr(self, "input_features", {}) or {}
        for v in features.values():
            if v.type == FeatureType.STATE:
                return v
        return None

    @classmethod
    def register_subclass(cls, name: str):
        """Decorator: @PreTrainedConfig.register_subclass("smolvla")"""
        def decorator(subclass):
            _REGISTRY[name] = subclass
            return subclass
        return decorator

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path: str, **kwargs):
        """Load config from a local dir or HuggingFace Hub repo."""
        path = Path(pretrained_name_or_path)
        if path.is_dir():
            config_path = path / "config.json"
        else:
            config_path = Path(
                hf_hub_download(pretrained_name_or_path, "config.json")
            )

        with open(config_path) as f:
            data = json.load(f)

        # find the right subclass from the "type" field
        cfg_type = data.get("type", "")
        subclass = _REGISTRY.get(cfg_type, cls)

        # filter to only known dataclass fields (ignore unknown hub metadata)
        import dataclasses
        known = {f.name for f in dataclasses.fields(subclass)}
        filtered = {k: v for k, v in data.items() if k in known}

        # Convert input_features / output_features dicts → PolicyFeature objects
        for feat_key in ("input_features", "output_features"):
            if feat_key in filtered and isinstance(filtered[feat_key], dict):
                converted = {}
                for k, v in filtered[feat_key].items():
                    if isinstance(v, dict):
                        converted[k] = PolicyFeature(
                            type=FeatureType(v["type"]),
                            shape=tuple(v["shape"]),
                        )
                    else:
                        converted[k] = v
                filtered[feat_key] = converted

        return subclass(**{**filtered, **kwargs})

    def save_pretrained(self, save_dir: str):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        def _default(o):
            if hasattr(o, "__dataclass_fields__"):
                return asdict(o)
            if hasattr(o, "value"):   # Enum
                return o.value
            return str(o)

        with open(save_dir / "config.json", "w") as f:
            json.dump(asdict(self), f, indent=2, default=_default)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
