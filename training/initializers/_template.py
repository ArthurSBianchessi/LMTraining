from typing import Protocol, runtime_checkable

from modeling.tokenizers._template import TokenizerTemplate
from modeling.models._template import ModelTemplate

@runtime_checkable
class InitializerTemplate(Protocol):
    """
    Protocol for model initializers.
    """
    def __init__(self, **kwargs):
        """
        Initialize the Initializer with specific arguments.
        """
        ...

    def __call__(self, model: ModelTemplate, tokenizer: TokenizerTemplate) -> None:
        """
        Apply initialization to the model.

        Args:
            model: The model to initialize.
            tokenizer: The tokenizer used with the model.
        """
        ...
