# -*- coding: utf-8 -*-

# ***************************************************
# * File        : llama2_7b_sentencepiece.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-03-30
# * Version     : 1.0.033002
# * Description : description
# * Link        : Google SentencePiece: https://github.com/google/sentencepiece
# *               Llama2 model weights and tokenizer vocabulary on huggingface hub: https://huggingface.co/meta-llama/Llama-2-7b
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
from pathlib import Path
try:
    from dotenv import find_dotenv, load_dotenv
except ImportError:  # python-dotenv is optional
    find_dotenv = None
    load_dotenv = None

from huggingface_hub import login, hf_hub_download
import sentencepiece as spm

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def _login_huggingface_hub():
    """
    login HuggingFace Hub
    """
    # load env variables
    if load_dotenv is not None:
        _ = load_dotenv(find_dotenv())
    # notebok
    # import getpass
    # os.environ["HF_HUB_ACCESS_TOKEN"] = getpass.get_pass()
    # script
    HF_HUB_ACCESS_TOKEN = os.getenv("HUGGINGFACE_HUB_API_KEY")
    # logger.info(f"HF_HUB_ACCESS_TOKEN: {HF_HUB_ACCESS_TOKEN}")

    login(token = HF_HUB_ACCESS_TOKEN)


def _download_tokenizer(model_path):
    """
    download llama2 tokenizer
    """
    tokenizer_file = hf_hub_download(
        repo_id = "meta-llama/Llama-2-7b",
        filename = "tokenizer.model",
        local_dir = model_path,
    )
    
    return tokenizer_file


class Llama27bTokenizer:

    def __init__(self, tokenizer_file_path=None):
        if tokenizer_file_path is None:
            tokenizer_file_path = Path(__file__).resolve().parents[3] / "downloaded_models/llama_model/Llama-2-7b/tokenizer.model"
        model_path = Path(tokenizer_file_path).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file {model_path} not found")
        sp = spm.SentencePieceProcessor()
        sp.load(str(model_path))
        self.tokenizer = sp
        self.n_vocab = self.tokenizer.vocab_size()
        self.eos_token_id = self.tokenizer.eos_id()
        self.pad_token_id = self.tokenizer.pad_id()
        if self.pad_token_id < 0:
            self.pad_token_id = self.eos_token_id

    def encode(self, text):
        return self.tokenizer.encode_as_ids(text)
    
    def decode(self, ids):
        # ids 是 int 列表，应使用 decode（等价于 decode_ids）；
        # decode_pieces 只接受 string pieces
        return self.tokenizer.decode(ids)




# 测试代码 main 函数
def main():
    # model path
    # model_path = "downloaded_models/llama_model/Llama-2-7b"
    # tokenizer_file_path = Path(model_path).joinpath("tokenizer.model")

    # login huggingface hub
    # _login_huggingface_hub()

    # download tokenizer model
    # _download_tokenizer(model_path)

    # tokenizer
    tokenizer = Llama27bTokenizer()

if __name__ == "__main__":
    main()
