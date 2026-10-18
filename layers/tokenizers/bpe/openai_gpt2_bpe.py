# Source: https://github.com/openai/gpt-2/blob/master/src/encoder.py
# License:
# Modified MIT License

# Software Copyright (c) 2019 OpenAI

# We don’t claim ownership of the content you create with GPT-2, so it is yours to do with as you please.
# We only ask that you use GPT-2 responsibly and clearly indicate your content was created using GPT-2.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# The above copyright notice and this permission notice need not be included
# with content created by the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.

import json
import regex as re
from pathlib import Path
from functools import lru_cache
import requests
from tqdm import tqdm
from typing import List


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]

    return dict(zip(bs, cs))


def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    
    return pairs


class Encoder:

    def __init__(self, encoder, bpe_merges, errors='replace'):
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should have added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word

        return word

    def encode(self, text):
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))

        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)

        return text


def get_encoder(model_name="gpt2_model", model_dir=Path("./layers/tokenizers/models")):
    # model files path
    model_file_path = model_dir.joinpath(model_name)
    
    # model file download
    from layers.tokenizers.bpe.vocab_download import download_vocab
    download_vocab(model_file_path=model_file_path, model_name="117M")
    
    # encoder.json file
    with open(model_file_path.joinpath('encoder.json'), 'r') as f:
        encoder = json.load(f)
    # vocab.bpe file
    with open(model_file_path.joinpath('vocab.bpe'), 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    
    # bpe merge
    bpe_merges = [
        tuple(merge_str.split()) 
        for merge_str in bpe_data.split('\n')[1:-1]
    ]

    return Encoder(encoder=encoder, bpe_merges=bpe_merges)


class OpenaiGPT2BPE:

    def __init__(self, tokenizer_model: str = "gpt2"):
        # model name
        model_name = "gpt2_model"
        # model path
        model_dir = Path("./layers/tokenizers/models")
        # tokenizer
        self.tokenizer = get_encoder(model_name=model_name, model_dir=model_dir)
    
    @property
    def n_vocab(self):
        return self.tokenizer.n_vocab

    def encode(self, text: str):
        """
        text encode to token IDs
        """
        token_ids = self.tokenizer.encode(text=text)

        return token_ids
    
    def decode(self, tokens: List):
        """
        token IDs decode to text
        """
        text = self.tokenizer.decode(tokens=tokens)
        
        return text




# 测试代码 main 函数
def main():
    import sys
    from pathlib import Path
    ROOT = str(Path.cwd())
    if ROOT not in sys.path:
        sys.path.append(ROOT)
    from utils.log_util import logger


    # tokenizer
    tokenizer = OpenaiGPT2BPE()
    # input_text
    input_text = "Hello, world. Is this-- a test?"
    # encode test
    input_text_token_ids = tokenizer.encode(text=input_text)
    logger.info(f"input_text_token_ids: {input_text_token_ids}")
    # decode test
    input_text_decoded = tokenizer.decode(tokens=input_text_token_ids)
    logger.info(f"input_text_decoded: {input_text_decoded}")

if __name__ == "__main__":
    main()
