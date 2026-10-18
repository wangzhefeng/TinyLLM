# -*- coding: utf-8 -*-

# ***************************************************
# * File        : qwen3_tokenizer_optimized.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-10-02
# * Version     : 1.0.100217
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import re
import warnings
warnings.filterwarnings("ignore")

from tokenizers import Tokenizer

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Qwen3Tokenizer:
    _SPECIALS = [
        "<|endoftext|>",
        "<|im_start|>", "<|im_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>", "<|box_end|>",
        "<|quad_start|>", "<|quad_end|>",
        "<|vision_start|>", "<|vision_end|>",
        "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
    ]
    _SPLIT_RE = re.compile(r"(<\|[^>]+?\|>)")

    def __init__(self, 
                 tokenizer_file_path="tokenizer-base.json",
                 apply_chat_template=False,
                 add_generation_prompt=False, 
                 add_thinking=False):
        # args
        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt = add_generation_prompt
        self.add_thinking = add_thinking
        # tokenizer file path
        tok_path = Path(tokenizer_file_path)
        if not tok_path.is_file():
            raise FileNotFoundError(f"Tokenizer file '{tok_path}' not found. Please allocate it's available.")
        # tokenizer init
        self._tok = Tokenizer.from_file(str(tok_path))
        # special token id
        self._special_to_id = {t: self._tok.token_to_id(t) for t in self._SPECIALS}
        # ------------------------------
        # Property::pad_token_id
        # ------------------------------
        self.pad_token_id = self._special_to_id.get("<|endoftext|>")
        # ------------------------------
        # Property::eos_token_id: Match HF behavior: chat model → <|im_end|>, base model → <|endoftext|>
        # ------------------------------
        fname = tok_path.name.lower()
        if "base" in fname and "reasoning" not in fname:
            eos_token = "<|endoftext|>"
        else:
            eos_token = "<|im_end|>"
        self.eos_token_id = self._special_to_id.get(eos_token)

    def encode(self, prompt, chat_wrapped=None):
        # chat wrapped arg
        if chat_wrapped is None:
            chat_wrapped = self.apply_chat_template
        
        # prompt is special token id
        stripped = prompt.strip()
        if stripped in self._special_to_id and "\n" not in stripped:
            return [self._special_to_id[stripped]]
        
        # chat wrapped
        if chat_wrapped:
            prompt = self._wrap_chat(prompt)
        # encode
        ids = []
        for part in filter(None, self._SPLIT_RE.split(prompt)):
            if part in self._special_to_id:
                ids.append(self._special_to_id[part])
            else:
                ids.extend(self._tok.encode(part).ids)
        return ids

    def decode(self, token_ids):
        return self._tok.decode(token_ids, skip_special_tokens=False)

    def _wrap_chat(self, user_msg):
        # user prompt
        s = f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        # add generation prompt
        if self.add_generation_prompt:
            s += "<|im_start|>assistant"
            # add thinking
            if self.add_thinking:
                s += "\n"  # insert no <think> tag, just a new line
            else:
                s += "\n<think>\n\n</think>\n\n"
        return s




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
