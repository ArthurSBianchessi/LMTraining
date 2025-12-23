from typing import Protocol, runtime_checkable

@runtime_checkable
class SchedulerTemplate(Protocol):
    def __init__(self, *args, **kwargs):
        pass
    
    def step(self):
        pass

    def state_dict(self) -> dict:
        pass

    def load_state_dict(self, state_dict: dict) -> None:
        pass
