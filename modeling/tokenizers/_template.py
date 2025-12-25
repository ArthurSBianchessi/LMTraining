from typing import Protocol, List, Union, runtime_checkable

@runtime_checkable
class Tokenizer(Protocol):
    def __init__(self, **kwargs):
        ...

    @property
    def vocab_size(self) -> int:
        ...

    def __call__(self, text: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        ...
