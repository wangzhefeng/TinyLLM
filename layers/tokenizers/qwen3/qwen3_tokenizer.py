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
from pathlib import Path
import re

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
                 tokenizer_file_path="tokenizer.json",
                 apply_chat_template=False,
                 add_generation_prompt=False,
                 add_thinking=False,
                 base_model=False):
        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt = add_generation_prompt
        self.add_thinking = add_thinking
        tok_path = Path(tokenizer_file_path).expanduser().resolve()
        if not tok_path.is_file():
            raise FileNotFoundError(f"Tokenizer file '{tok_path}' not found")
        self._tok = Tokenizer.from_file(str(tok_path))
        self._special_to_id = {t: self._tok.token_to_id(t) for t in self._SPECIALS}
        self.n_vocab = self._tok.get_vocab_size()
        self.pad_token_id = self._special_to_id.get("<|endoftext|>")
        eos_token = "<|endoftext|>" if base_model else "<|im_end|>"
        self.eos_token_id = self._special_to_id.get(eos_token)
        if self.eos_token_id is None:
            raise ValueError(f"Tokenizer does not define required eos token {eos_token}")

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
        from layers.tokenizers.chat import format_qwen3_chat

        return format_qwen3_chat(
            user_msg,
            add_generation_prompt=self.add_generation_prompt,
            add_thinking=self.add_thinking,
        )




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
