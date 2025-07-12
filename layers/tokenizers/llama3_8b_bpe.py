# -*- coding: utf-8 -*-

# ***************************************************
# * File        : tokenizer_bpe.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-23
# * Version     : 0.1.022314
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = [
    "Tokenizer",
]

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)


from dotenv import find_dotenv, load_dotenv
from huggingface_hub import login, hf_hub_download
import tiktoken
from tiktoken.load import load_tiktoken_bpe

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def _login_huggingface_hub():
    """
    login HuggingFace Hub
    """
    # load env variables
    _ = load_dotenv(find_dotenv())
    # notebok
    # import getpass
    # os.environ["HF_HUB_ACCESS_TOKEN"] = getpass.get_pass()
    # script
    HF_HUB_ACCESS_TOKEN = os.getenv("HUGGINGFACE_HUB_API_KEY")
    # logger.info(f"HF_HUB_ACCESS_TOKEN: {HF_HUB_ACCESS_TOKEN}")

    login(token = HF_HUB_ACCESS_TOKEN)


def _download_tokenzier(model_path):
    """
    download llama2 tokenizer
    """
    tokenizer_file = hf_hub_download(
        repo_id="meta-llama/Meta-Llama-3-8B",
        filename="original/tokenizer.model",
        local_dir = model_path,
    )
    
    return tokenizer_file


class Llama38bTokenizer:
    
    def __init__(self, special_token_version = "v1"):
        """
        Args:
            model_path (_type_): 分词器模型的路径
        """
        # ------------------------------
        # 分词器模型的路径
        # ------------------------------
        model_path = "downloaded_models/llama_model/Meta-Llama-3-8B/original/tokenizer.model"
        assert os.path.isfile(model_path), f"Model file {model_path} not found"
        # ------------------------------
        # 加载 BPE 模型(实际是一个字典)
        # ------------------------------
        # 一个字典，子词(bytes 类型，用 utf-8 解码)-rank(id) 对，128000 个词，
        # 不包含上面的 256 个特殊 token（所以最终模型的总词典大小是 128256）
        # 其中 rank 值是从 0 递增的序列，用于决定子词单元合并的优先顺序，
        # 优先级越高的会优先合并，因此这里的名字是 mergeable ranks 而非 BPE 或字典等类似的名字
        # 没把特殊 token 加到字典里应该是出于灵活性考虑，
        # 便于面对不同模型架构或任务有不同特殊 token 时添加特定的 token，而且保持字典大小不变
        # mergeable ranks
        mergeable_ranks = load_tiktoken_bpe(model_path)
        # ------------------------------
        # 常规词典外的特殊 token
        # ------------------------------
        # 在"Meta-Llama-3-8B/"路径下的 'tokenizer.json' 和
        # 'tokenizer_config.json'的 added_tokens 字段下都有这些特殊 token
        if special_token_version == "v1":
            self.special_tokens = {
                "<|begin_of_text|>": 128000,
                "<|end_of_text|>": 128001,
                "<|start_header_id|>": 128006,
                "<|end_header_id|>": 128007,
                "<|eot_id|>": 128009,
            }
            self.special_tokens.update({
                f"<|reserved_{i}|>": 128002 + i 
                for i in range(256) 
                if (128002 + i) not in self.special_tokens.values()
            })
        else:
            self.special_tokens = [
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
                f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)
            ]
            self.special_tokens = {
                token: len(mergeable_ranks) + i 
                for i, token in enumerate(self.special_tokens)
            }
        # logger.info(f"special_tokens length: {len(self.special_tokens)}")
        # logger.info(f"special_tokens: {self.special_tokens}")
        # ------------------------------
        # 创建一个文本编码器对象
        # ------------------------------
        # 其中 pat_str 大致分为三个模型：1:带缩写的单词和单词, 2:中文片段, 3:1-3位的数字和其它特殊字符
        self.model = tiktoken.Encoding(
            # 编码器名称，便于调试和日志记录使用的不同编码器
            name = Path(model_path).name,
            # 用于初步的粗分割文本为token序列的正则表达式
            pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            # 传入加载的 BPE 模型
            mergeable_ranks = mergeable_ranks,
            # 添加特殊 token-id 对的字典
            special_tokens = self.special_tokens,
        )
    
    def encode(self, text, bos = False, eos = False, allowed_special = set(), disallowed_special = ()):
        # begin of text tokens
        if bos:
            tokens = [self.special_tokens["<|begin_of_text|>"]]
        else:
            tokens = []
        # tokens
        tokens += self.model.encode(
            text, 
            allowed_special = allowed_special, 
            disallowed_special = disallowed_special
        )
        # end of text tokens
        if eos:
            tokens.append(self.special_tokens["<|end_of_text|>"])
        
        return tokens
    
    def decode(self, tokens):
        return self.model.decode(tokens)




# 测试代码 main 函数
def main():
    # model path
    # model_path = "downloaded_models/llama_model/Llama-3-8b"
    # tokenizer_file_path = Path(model_path).joinpath("tokenizer.model")

    # login huggingface hub
    # _login_huggingface_hub()

    # download tokenizer model
    # _download_tokenzier(model_path)
    
    # 加载 tokenizer
    tokenizer = Llama38bTokenizer(special_token_version="v2")

    """
    # 下面是一个案例测试，来测试 pat_str 粗分割和 tokenizer 细分割的效果和区别
    # pat_str 的正则只是提供了一个初步的分割，一些长句子或中文等不会分割，
    # 会在 tokenizer 中进一步基于 BPE 算法进行细化分割 

    # 1.测试文本
    text = "Hello world! It's a test. 这是一个测试. alongwords. a long words. 123 456 789."
    logger.info(f"原始字符串:, \n{text}")

    # 2.使用正则表达式分割字符串
    # 创建正则: 由于 pat_str 中用到了 Unicode 的一些语法，如 \p{L}，所以不能用 re 库
    import regex
    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    pattern = regex.compile(pat_str)
    re_tokens = pattern.findall(text)
    logger.info(f"正则分割结果:, \n{re_tokens}")

    # 3.使用 tokenizer 分割字符串
    merge_tokens_id = tokenizer.encode(text)
    # 将 tokenizer 分割结果的 id 序列转换为实际的子词序列
    merge_tokens = [tokenizer.decode([i]) for i in merge_tokens_id] 
    logger.info(f"tokenizer 分割结果:, \n{merge_tokens}")
    logger.info(f"tokenizer 分割结果id:, \n{list(zip(merge_tokens, merge_tokens_id))}")

    # 4. 结果输出
    # 从结果将会看到所有单词的前缀空格都被保留了下来，而非单独一个空格 token 或将其删除，
    # 有利于模型正确理解单词间的边界信息，如例子中的 alongwords
    """

if __name__ == "__main__":
    main()
