"""Stub for RTCProcessor — used only during online/real-time inference."""

import torch


class RTCProcessor:
    """
    Real-Time Control Processor stub.

    The full implementation handles latency-aware action smoothing during
    deployment.  For offline training and evaluation this class is never
    instantiated — it's only referenced as a type.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            "RTCProcessor is only needed for real-time robot deployment, "
            "not for offline training."
        )
