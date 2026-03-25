"""
Stub processor module.

processor_smolvla.py imports from here, but we do NOT use that file
during training — we build the batch ourselves in dataset.py.
These stubs exist only to satisfy any top-level imports.
"""


class PolicyAction:
    pass


class PolicyProcessorPipeline:
    pass


class ProcessorStepRegistry:
    pass


class AddBatchDimensionProcessorStep:
    pass


class ComplementaryDataProcessorStep:
    pass


class DeviceProcessorStep:
    pass


class NormalizerProcessorStep:
    pass


class UnnormalizerProcessorStep:
    pass


class RenameObservationsProcessorStep:
    pass


class TokenizerProcessorStep:
    pass
