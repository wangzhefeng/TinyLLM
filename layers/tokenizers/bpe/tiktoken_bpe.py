import tiktoken

__all__ = ["TiktokenBPE"]


class TiktokenBPE:
    def __init__(self, tokenizer_model: str = "gpt2"):
        self.tokenizer = tiktoken.get_encoding(tokenizer_model)
        self.n_vocab = self.tokenizer.n_vocab
        self.eos_token_id = self.tokenizer.eot_token
        self.pad_token_id = self.tokenizer.eot_token

    def encode(self, text: str, allowed_special=None, **kwargs):
        if allowed_special is None:
            allowed_special = {"<|endoftext|>"}
        return self.tokenizer.encode(text=text, allowed_special=allowed_special, **kwargs)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens=tokens)
