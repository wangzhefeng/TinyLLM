# -*- coding: utf-8 -*-

# ***************************************************
# * File        : tokenizer_bpe.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-28
# * Version     : 1.0.102801
# * Description : 分词器章节示例：词级 SimpleTokenizer 的编码-解码往返，
# *               与 BPE 子词分词的对照（tiktoken gpt2）。学习路线图阶段 1。
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

from layers.tokenizers.simple.simple_tokenizer import SimpleTokenizer
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def word_level_tokenizer():
    """词级分词：文本 -> 词表 -> token id 序列 -> 还原。"""
    text = "A cat eats a fish. The cat is happy."
    tokenizer = SimpleTokenizer(text)
    logger.info(f"vocab size: {len(tokenizer.str_to_int)}")
    ids = tokenizer.encode("A cat eats")
    logger.info(f"encode('A cat eats') -> {ids}")
    logger.info(f"decode({ids}) -> {tokenizer.decode(ids)}")
    # 未见过的词在词级方案里只能映射为 <|unk|>，信息丢失（BPE 的动机）
    unk_ids = tokenizer.encode("A dogeats")
    logger.info(f"encode('A dogeats') -> {tokenizer.decode(unk_ids)!r}"
                "（未登录词被吞成 <|unk|>，BPE 的动机）")


def bpe_tokenizer():
    """BPE 子词分词：常见词整体成 token，罕见词拆成子词/字节。"""
    try:
        import tiktoken
        tokenizer = tiktoken.get_encoding("gpt2")
    except Exception as e:  # 未联网下载过编码文件时跳过
        logger.info(f"tiktoken gpt2 编码不可用（{e}），跳过 BPE 演示")
        return
    for text in ("Every effort moves", "unbelievable AntiDisEstablishmentArianIsm"):
        ids = tokenizer.encode(text)
        tokens = [tokenizer.decode([i]) for i in ids]
        logger.info(f"{text!r} -> {ids}")
        logger.info(f"切分结果: {tokens}")


# 测试代码 main 函数
def main():
    logger.info(f"{'-'*20} 词级 SimpleTokenizer {'-'*20}")
    word_level_tokenizer()
    logger.info(f"{'-'*20} BPE（tiktoken gpt2） {'-'*20}")
    bpe_tokenizer()

if __name__ == "__main__":
    main()
