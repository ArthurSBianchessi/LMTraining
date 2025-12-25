from typing import List
import fnmatch
import torch.nn as nn
from modeling.models._template import ModelTemplate
from modeling.tokenizers._template import TokenizerTemplate
from training.initializers._template import InitializerTemplate

class ZeroInitializer:
    """
    Initializer that zeroes out parameters matching any of the provided glob patterns.
    """
    def __init__(self, patterns: List[str]):
        """
        Args:
            patterns: A list of glob patterns. Any parameter whose name matches
                      one of these patterns will be zero-initialized.
                      Example: ['*bias*', 'lm_head.weight', '*.c_proj.weight']
        """
        self.patterns = patterns

    def __call__(self, model: ModelTemplate, tokenizer: TokenizerTemplate) -> None:
        """
        Zero out matching parameters.
        """
        if not self.patterns:
            return

        for name, param in model.named_parameters():
            for pattern in self.patterns:
                if fnmatch.fnmatch(name, pattern):
                    nn.init.zeros_(param)
                    # Break after finding first match to avoid redundant work
                    break
