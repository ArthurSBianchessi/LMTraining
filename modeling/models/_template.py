from typing import Protocol, runtime_checkable, Any

@runtime_checkable
class ModelTemplate(Protocol):
    """
    Protocol for models.

    It is recommended (but not required) that implementations inherit from
    `torch.nn.Module` to leverage PyTorch's infrastructure for parameters,
    gradients, and device management.
    """
    def __init__(self, **kwargs):
        """
        Initialize the model with specific arguments.
        """
        ...

    def __call__(self, *args, **kwargs) -> Any:
        """
        Forward pass of the model.

        If inheriting from `torch.nn.Module`, implement `forward` instead of `__call__`.
        """
        ...
