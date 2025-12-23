import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class CrossEntropy(nn.Module):
    def __init__(self, target_key: str = "labels", ignore_index: int = -100):
        super().__init__()
        self.target_key = target_key
        self.ignore_index = ignore_index

    def forward(self, data: Dict[str, Any], prediction: torch.Tensor) -> Dict[str, Any]:
        targets = data[self.target_key]
        loss = F.cross_entropy(prediction, targets, ignore_index=self.ignore_index)
        return {"loss": loss}
