# mini_clip/models/tokenizer.py

import torch
from transformers import AutoTokenizer

class HFTokenizer:
    """
    A thin wrapper around a Hugging Face tokenizer.
    Example uses the 'gpt2' tokenizer, but you can specify
    any model checkpoint that includes a tokenizer in Hugging Face.
    """
    def __init__(self, pretrained_name="gpt2", max_length=64):
        """
        Args:
            pretrained_name (str): Name or path of a pretrained tokenizer.
            max_length (int): Maximum length for token sequences.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        self.tokenizer.model_max_length = max_length
        self.max_length = max_length

        if not self.tokenizer.pad_token:
            # Some GPT-2 tokenizers do not have a PAD token. We can assign to EOS
            # or a new token, if desired.
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, text: str):
        """
        Tokenize input text and return a list of token IDs.
        """
        # return_tensors='pt' can be used here if you want a torch tensor
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True,     # includes BOS/EOS tokens etc.
            padding="max_length",       
            return_tensors=None
        )
        token_ids = encoding["input_ids"]
        return token_ids
