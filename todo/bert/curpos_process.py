# -*- coding: utf-8 -*-

# ***************************************************
# * File        : curpos_process.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-01
# * Version     : 1.0.010122
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tokenizers import BertTokenizerFast

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def encode_with_truncation(examples, tokenizer):
    """
    使用词元分析对句子进行处理并截断的映射函数

    Args:
        example (_type_): _description_
    """
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_special_tokens_mask=True
    )


def encode_without_truncation(examples, tokenizer):
    """
    使用词元分析对句子进行处理但是不截断的映射函数

    Args:
        examples (_type_): _description_
    """
    return tokenizer(
        examples["text"],
        return_special_tokens_mask=True
    )




# 测试代码 main 函数
def main():
    project_dir = "/Users/wangzf/llm_proj/tiny_llm//bert/"
    # tokenzier 模型路径
    model_path = os.path.join(project_dir, "pretrained-bert")
    # 当词元分析器进行训练和配置时，将其装载到 BertTokenizerFast 中
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    
    # 是否将长样本截断
    truncate_longer_samples = False
    
    # 编码函数将用依赖于 truncate_longer_samples 变量
    encode = encode_with_truncation if truncate_longer_samples else encode_without_truncation

    # 对训练数据集进行分词处理
    

if __name__ == "__main__":
    main()
