from typing import Any, Iterator, Protocol, runtime_checkable, Union, Dict

try:
    from torch.utils.data import Dataset, IterableDataset
except ImportError:
    # Fallback to Any if torch is not available, though strongly recommended
    Dataset = Any
    IterableDataset = Any

@runtime_checkable
class MapDatasetTemplate(Protocol):
    """
    Protocol describing a map-style dataset (implements __getitem__ and __len__).
    
    Requirements:
    - Must have a `tokenizer` attribute.
    - `__getitem__` must return a dictionary containing at least an "input" key.
    
    It is recommended to inherit from `torch.utils.data.Dataset`.
    """
    tokenizer: Any

    def __init__(self, tokenizer: Any, *args, **kwargs):
        ...

    def __getitem__(self, index: Any) -> Dict[str, Any]:
        ...

    def __len__(self) -> int:
        ...

@runtime_checkable
class IterableDatasetTemplate(Protocol):
    """
    Protocol describing an iterable-style dataset (implements __iter__).
    
    Requirements:
    - Must have a `tokenizer` attribute.
    - `__iter__` must yield dictionaries containing at least an "input" key.
    
    It is recommended to inherit from `torch.utils.data.IterableDataset`.
    """
    tokenizer: Any

    def __init__(self, tokenizer: Any, *args, **kwargs):
        ...

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        ...

DatasetTemplate = Union[MapDatasetTemplate, IterableDatasetTemplate]
"""
Type alias for a Dataset, which can be either a map-style dataset or an iterable dataset.
Compatible with `torch.utils.data.Dataset` and `torch.utils.data.IterableDataset`.
"""
