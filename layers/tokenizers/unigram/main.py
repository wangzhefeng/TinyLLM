# -*- coding: utf-8 -*-

# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102401
# * Description : Unigram（SentencePiece 风格）分词器小演示：
# *               在 ~10 句小语料上训练 UnigramTokenizer，
# *               展示 encode / decode / save / load。
# * Link        : link
# * Requirement : 无第三方依赖
# ***************************************************

__all__ = []

# python libraries
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger

from layers.tokenizers.unigram.unigram import UnigramTokenizer




# 测试代码 main 函数
def main():
    # ~10 句小语料
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "the dog barks at the fox",
        "a quick brown dog runs fast",
        "the lazy dog sleeps all day",
        "foxes are quick and clever animals",
        "the brown fox is quick",
        "dogs and foxes are animals",
        "the cat sleeps on the mat",
        "a lazy cat sleeps all day",
        "the quick dog jumps over the lazy cat",
    ]

    # 训练
    tokenizer = UnigramTokenizer(vocab_size=100)
    tokenizer.train(corpus)
    logger.info(f"tokenizer.n_vocab: {tokenizer.n_vocab}")

    # encode / decode 展示
    text = "the quick brown fox jumps over the lazy dog"
    ids = tokenizer.encode(text)
    logger.info(f"text: {text!r}")
    logger.info(f"ids: {ids}")
    logger.info(f"tokens: {[tokenizer.id_to_token(i) for i in ids]}")
    logger.info(f"decoded: {tokenizer.decode(ids)!r}")

    # 未见过的字符走 unk，不抛异常
    unseen_ids = tokenizer.encode("hello world!!!")
    logger.info(f"unseen text ids: {unseen_ids}")

    # save / load 往返
    save_path = Path("./layers/tokenizers/models/unigram_demo.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(save_path)
    loaded = UnigramTokenizer.load(save_path)
    logger.info(f"reload encode match: {loaded.encode(text) == ids}")

if __name__ == "__main__":
    main()
