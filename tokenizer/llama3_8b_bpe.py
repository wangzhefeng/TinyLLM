# -*- coding: utf-8 -*-

# ***************************************************
# * File        : tokenizer_bpe.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-23
# * Version     : 0.1.022314
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = [
    "llama3_8b_tokenizer",
]

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from pathlib import Path

import tiktoken
from tiktoken.load import load_tiktoken_bpe

from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def llama3_8b_tokenizer():
    # ------------------------------
    # 分词器模型的路径
    # ------------------------------
    tokenizer_path = os.path.join(ROOT, r"tokenizer\Meta-Llama-3-8B\original\tokenizer.model")
    # logger.info(f"tokenizer_path: {tokenizer_path}")
    # ------------------------------
    # 常规词典外的特殊 token
    # ------------------------------
    # 在"Meta-Llama-3-8B/"路径下的'tokenizer.json'和
    # 'tokenizer_config.json'的added_tokens字段下都有这些特殊token
    special_tokens = [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|reserved_special_token_0|>",  # 保留了从 0 到 250 的特殊 token
        "<|reserved_special_token_1|>",
        "<|reserved_special_token_2|>",
        "<|reserved_special_token_3|>",
        "<|start_header_id|>",  # 头部信息的开始，用于标记包裹结构化数据的头部信息，如元数据
        "<|end_header_id|>",  # 头部信息的结束，用于标记包裹结构化数据的头部信息，如元数据
        "<|reserved_special_token_4|>",
        "<|eot_id|>",  # end of turn, 对轮对话里标记当前轮次对话的结束
    ] + [
        f"<|reserved_special_token_{i}|>"
        for i in range(5, 256 - 5)
    ]
    logger.info(f"special_tokens length: {len(special_tokens)}")
    # ------------------------------
    # 加载 BPE 模型(实际是一个字典)
    # ------------------------------
    # 一个字典，子词(bytes 类型，用 utf-8 解码)-rank(id) 对，128000 个词，
    # 不包含上面的 256 个特殊 token（所以最终模型的总词典大小是 128256）
    # 其中 rank 值是从 0 递增的序列，用于决定子词单元合并的优先顺序，
    # 优先级越高的会优先合并，因此这里的名字是 mergeable ranks 而非 BPE 或字典等类似的名字
    # 没把特殊 token 加到字典里应该是出于灵活性考虑，
    # 便于面对不同模型架构或任务有不同特殊 token 时添加特定的 token，而且保持字典大小不变
    mergealbe_ranks = load_tiktoken_bpe(tokenizer_path)
    logger.info(f"mergealbe_ranks length: {len(mergealbe_ranks)}")
    # ------------------------------
    # 创建一个文本编码器对象
    # ------------------------------
    # 其中 pat_str 大致分为三个模型：1:带缩写的单词和单词, 2:中文片段, 3:1-3位的数字和其它特殊字符
    tokenizer = tiktoken.Encoding(
        name = Path(tokenizer_path).name,  # 编码器名称，便于调试和日志记录使用的不同编码器
        pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",  # 用于初步的粗分割文本为token序列的正则表达式
        mergeable_ranks = mergealbe_ranks,  # 传入加载的 BPE 模型
        special_tokens = {
            token: len(mergealbe_ranks) + i 
            for i, token in enumerate(special_tokens)
        }  # 添加特殊 token-id 对的字典
    )
    # logger.info(f"{tokenizer.decode(tokenizer.encode('create tokenizer successed!'))}")

    return tokenizer




# 测试代码 main 函数
def main():
    # 下面是一个案例测试，来测试 pat_str 粗分割和 tokenizer 细分割的效果和区别
    # pat_str 的正则只是提供了一个初步的分割，一些长句子或中文等不会分割，
    # 会在 tokenizer 中进一步基于 BPE 算法进行细化分割 

    # ------------------------------
    # 加载 tokenizer
    # ------------------------------
    tokenizer = llama3_8b_tokenizer()
    # ------------------------------
    # 创建正则
    # ------------------------------
    # 由于 pat_str 中用到了 Unicode 的一些语法，如 \p{L}，所以不能用 re 库
    import regex
    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    pattern = regex.compile(pat_str)
    # ------------------------------
    # 文本切分
    # ------------------------------
    # 1.测试文本
    text = "Hello world! It's a test. 这是一个测试. alongwords. a long words. 123 456 789."
    # 2.使用正则表达式分割字符串
    re_tokens = pattern.findall(text)
    # 3.使用 tokenizer 分割字符串
    merge_tokens_id = tokenizer.encode(text)
    # 3.1 将 tokenizer 分割结果的 id 序列转换为实际的子词序列
    merge_tokens = [tokenizer.decode([i]) for i in merge_tokens_id] 
    # ------------------------------
    # 结果输出
    # ------------------------------
    # 从结果将会看到所有单词的前缀空格都被保留了下来，而非单独一个空格 token 或将其删除，
    # 有利于模型正确理解单词间的边界信息，如例子中的 alongwords
    logger.info(f"原始字符串:, \n{text}")
    logger.info(f"正则分割结果:, \n{re_tokens}")
    logger.info(f"tokenizer分割结果:, \n{merge_tokens}")
    logger.info(f"tokenizer分割结果id:, \n{list(zip(merge_tokens, merge_tokens_id))}")

if __name__ == "__main__":
    main()
