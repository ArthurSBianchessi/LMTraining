import math
import torch.nn as nn
from modeling.models._template import ModelTemplate
from modeling.tokenizers._template import TokenizerTemplate
from modeling.initializers._template import InitializerTemplate

class NanoChatInitializer:
    """
    Initializer that replicates the initialization strategy from Andrej Karpathy's NanoChat,
    but with configurable names for projection layers and the language model head.
    
    Strategy:
    - Linear layers: Normal init with mean=0.0, std = 1.0 / sqrt(fan_in) * min(1.0, sqrt(fan_out / fan_in))
    - Embedding layers: Normal init with mean=0.0, std=1.0
    - Biases: Zero
    - Special zeroing:
        - Weights of the language model head (identified by `head_name`).
        - Weights of any Linear layer whose name contains `projection_subword`.
    """
    def __init__(self, projection_subword: str = "out", head_name: str = "lm_head"):
        """
        Args:
            projection_subword: Substring to identify projection layers that should be zero-initialized.
                                Defaults to "out".
            head_name: Name of the language model head attribute in the model.
                       Defaults to "lm_head".
        """
        self.projection_subword = projection_subword
        self.head_name = head_name

    def __call__(self, model: ModelTemplate, tokenizer: TokenizerTemplate) -> None:
        """
        Initialize the model weights.
        """
        # Apply base initialization
        model.apply(self._init_weights)
        
        # Apply special zeroing
        # Zero out classifier weights if they exist
        if hasattr(model, self.head_name):
            head = getattr(model, self.head_name)
            if isinstance(head, nn.Linear):
                 nn.init.zeros_(head.weight)
             
        # Zero out weights of layers matching the projection subword
        for name, module in model.named_modules():
            if self.projection_subword in name and isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            # NanoChat: std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=1.0)
