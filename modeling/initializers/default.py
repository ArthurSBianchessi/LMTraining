from modeling.models._template import ModelTemplate
from modeling.tokenizers._template import TokenizerTemplate
from modeling.initializers._template import InitializerTemplate

class DefaultInitializer:
    """
    Default initializer that performs no initialization (identity operation).
    """
    def __init__(self):
        pass

    def __call__(self, model: ModelTemplate, tokenizer: TokenizerTemplate) -> None:
        pass
