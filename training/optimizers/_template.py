from typing import Protocol, runtime_checkable

@runtime_checkable
class OptimizerTemplate(Protocol):
    def __init__(self, *args, **kwargs):
        pass
    def step(self):
        pass