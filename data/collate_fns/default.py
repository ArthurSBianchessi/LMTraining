from typing import Any, Dict, List
import torch
from torch.utils.data import default_collate

class DefaultCollateFn:
    """
    Default collate function that uses `torch.utils.data.default_collate`.
    """
    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collates a list of samples using PyTorch's default_collate.
        
        Args:
            samples: A list of dictionaries.
            
        Returns:
            A dictionary containing the batched data.
        """
        return default_collate(samples)
