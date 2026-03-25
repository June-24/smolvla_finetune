"""Constants used throughout SmolVLA source code."""

import os
from pathlib import Path

# Observation keys
OBS_STR = "observation"
OBS_PREFIX = "observation."
OBS_ENV_STATE = "observation.environment_state"
OBS_STATE = "observation.state"
OBS_IMAGE = "observation.image"
OBS_IMAGES = "observation.images"
OBS_LANGUAGE = "observation.language"
OBS_LANGUAGE_TOKENS = "observation.language_tokens"
OBS_LANGUAGE_ATTENTION_MASK = "observation.language_attention_mask"
OBS_LANGUAGE_SUBTASK = "observation.language_subtask"
OBS_LANGUAGE_SUBTASK_TOKENS = "observation.language_subtask_tokens"
OBS_LANGUAGE_SUBTASK_ATTENTION_MASK = "observation.language_subtask_attention_mask"

# Action keys
ACTION = "action"
ACTION_PREFIX = "action."
ACTION_TOKENS = "action_tokens"
ACTION_TOKEN_MASK = "action_token_mask"

# Other keys
REWARD = "reward"
TRUNCATED = "truncated"
DONE = "done"
INFO = "info"
ROBOTS = "robots"
TELEOPERATORS = "teleoperators"

# Checkpoint keys
CHECKPOINTS_DIR = "checkpoints"
LAST_CHECKPOINT_LINK = "last"
PRETRAINED_MODEL_DIR = "pretrained_model"
TRAINING_STATE_DIR = "training_state"
RNG_STATE = "rng_state"
TRAINING_STEP = "training_step"
OPTIMIZER_STATE = "optimizer_state"
OPTIMIZER_PARAM_GROUPS = "optimizer_param_groups"
SCHEDULER_STATE = "scheduler_state"

# Processor keys
POLICY_PREPROCESSOR_DEFAULT_NAME = "policy_preprocessor"
POLICY_POSTPROCESSOR_DEFAULT_NAME = "policy_postprocessor"

# HF home
HF_LEROBOT_HOME = Path(os.getenv("HF_LEROBOT_HOME", Path.home() / ".cache" / "huggingface" / "lerobot"))
HF_LEROBOT_CALIBRATION = Path(
    os.getenv("HF_LEROBOT_CALIBRATION", HF_LEROBOT_HOME / "calibration")
)
