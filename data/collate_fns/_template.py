from typing import Any, List, Protocol, runtime_checkable, Dict, Union

@runtime_checkable
class CollateFnTemplate(Protocol):
    """
    Protocol describing a collate function.
    
    A collate function takes a list of samples (dictionaries) and merges them into a batch (dictionary of tensors).
    """
    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collates a list of samples into a batch.
        
        Args:
            samples: A list of dictionaries, where each dictionary represents a sample.
            
        Returns:
            A dictionary containing the batched data.
        """
        ...
