"""Stub converters — only used by processor_smolvla.py which we skip."""


def policy_action_to_transition(*args, **kwargs):
    raise NotImplementedError("processor converters not used during training")


def transition_to_policy_action(*args, **kwargs):
    raise NotImplementedError("processor converters not used during training")
