"""Stub for lerobot.utils.hub — HubMixin used by PreTrainedPolicy."""


class HubMixin:
    """Minimal mixin — actual push_to_hub not needed for local training."""

    def push_to_hub(self, *args, **kwargs):
        raise NotImplementedError("push_to_hub not implemented in standalone stub")
