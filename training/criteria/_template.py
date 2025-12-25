from typing import Any, Dict, Protocol, runtime_checkable

@runtime_checkable
class Criterion(Protocol):
    """
    Protocol describing the interface for a Criterion (loss function).

    It is recommended (but not required) that implementations inherit from
    `torch.nn.Module` to leverage PyTorch's infrastructure for keys,
    device movement, and parameter management.
    """

    # If inheriting from `torch.nn.Module`, you should implement `forward` instead
    # of `__call__`. `nn.Module.__call__` will automatically invoke `forward`
    # and handle hooks.
    def __call__(self, data: Dict[str, Any], prediction: Any, mode: str) -> Dict[str, Any]:
        """
        Compute the loss given the input data and the model's prediction.
        
        Args:
            data: The full batch of data, typically a dictionary.
            prediction: The output of the model (predictions).
            mode: The current execution mode (e.g., "train", "val", "test").

        Returns:
            A dictionary containing:
                - "loss": The computed loss scalar (Tensor).
                - "logs": A dictionary of metrics to log (e.g., {"loss": 0.5}).
        """
        ...
