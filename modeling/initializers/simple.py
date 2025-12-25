import math
import torch.nn as nn
from modeling.models._template import ModelTemplate
from modeling.tokenizers._template import TokenizerTemplate
from modeling.initializers._template import InitializerTemplate

class SimpleInitializer:
    """
    Simple initialier that initializes weights with std = 1.0 / sqrt(fan_in).
    Biases are initialized to zero.
    """
    def __init__(self):
        pass

    def __call__(self, model: ModelTemplate, tokenizer: TokenizerTemplate) -> None:
        """
        Initialize the model weights.
        """
        model.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=1.0)
