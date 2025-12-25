from typing import List, Union
from transformers import AutoTokenizer
from ._template import Tokenizer

class HuggingFaceTokenizer:
    def __init__(self, pretrained_model_name_or_path: str, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def eot_token_id(self) -> int:
        if self.tokenizer.eos_token_id is not None:
            return self.tokenizer.eos_token_id
        raise ValueError("Tokenizer does not have an EOS token.")

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def __call__(self, text: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        return self.tokenizer(text)["input_ids"]
