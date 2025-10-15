# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_provider.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-10-14
# * Version     : 1.0.101417
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
import random
from tqdm import tqdm
from typing import List

import pandas as pd

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class DataProcessor:
    def __init__(self, 
                 data_dir: str="./dataset/THUCNews", 
                 output_text_file: str="THUCNews.csv", 
                 output_task_file: str="task2.csv", 
                 index_2_word_file: str="index_2_word.txt"):
        self.data_dir = Path(data_dir)
        # 原始语料处理后的 CSV 文件
        self.output_text_file_path = self.data_dir.joinpath(output_text_file)
        # 构建 task2 数据集
        self.output_task_file_path = self.data_dir.joinpath(output_task_file)
        # 构建 tokenizer 数据文件
        self.index_2_word_file_path = self.data_dir.joinpath(index_2_word_file)
    
    def load_data(self):
        if not self.output_text_file_path.exists():
            # data source
            text_data_dir = self.data_dir.joinpath("text_data")
            # data create
            data = []
            for file_path in text_data_dir.glob("*.txt"):
                # field id
                file_id = int(file_path.name.split(".")[0])
                # field content: text data read
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read().strip()
                # text data collect
                data.append({"id": file_id, "content": content})
            # text data to dataframe
            df = pd.DataFrame(data)
            df = df.sort_values(by="id").reset_index(drop=True)
            df.to_csv(self.output_text_file_path, encoding="utf-8", index=False)
            # text data content list
            all_text = df["content"].tolist()
            return all_text
        else:
            # text data read
            all_data = pd.read_csv(self.output_text_file_path)
            # text data content list
            all_text = all_data["content"].tolist()
            return all_text

    def __split_text(self, text):
        """
        责将每个文本内容按照指定的标点符号（如中文的逗号、顿号、冒号、
        分号、句号、问号）进行分割，形成一系列的句子或短语。

        Args:
            text (_type_): _description_

        Returns:
            _type_: _description_
        """
        pattern = r"[，、：；。？]"
        sp_text = re.split(pattern, text)
        new_sp_text = self.__resplit_text(sp_text)
        
        return new_sp_text

    def __resplit_text(self, text_list):
        """
        对分割后的文本进行随机合并或单独保留，以形成新的文本片段列表。
        这个过程包括了一定的随机性，这里的随机性参考论文内容。
            - 在每个输入序列中，随机选择 15% 的单词进行遮蔽
            - 对于被选择的每个单词，有 80% 的概率将其替换为 `[MASK]` 标记
            - 有 10% 的概率将其替换为一个随机选择的单词
            - 有 10% 的概率保持单词不变
        """
        result = []
        sentence = ""
        for text in text_list:
            if len(text) < 3:
                continue

            if sentence == "":
                if random.random() < 0.2:
                    result.append(text + "。")
                    continue

            if len(sentence) < 30 or random.random() < 0.2:
                sentence += text + "，"
            else:
                result.append(sentence[:-1] + "。")
                sentence = text
        
        return result

    def __build_neg_pos_data(self, text_list):
        """
        生成正负样本对。对于给定的文本列表，它首先创建一个正样本对（相邻的两个文本），
        然后为每个正样本创建一个负样本对（其中一个文本与列表中随机选择的另一个文本配对）。
        正负样本的标签分别为1和0。第一个文本片段（句子A）使用A嵌入，第二个文本片段（句子B）使用B嵌入。
        50%的情况下，句子B是句子A之后的实际下文，即它们在原始文本中是连续的。另外50%的情况下，
        句子B是从语料库中随机选择的一个句子，与句子A没有实际的上下文关系。
        """
        all_text1, all_text2 = [], []
        all_label = []
        for tidx, text in enumerate(text_list):
            if tidx == len(text_list) - 1:
                break
            # pos data
            all_text1.append(text)
            all_text2.append(text_list[tidx + 1])
            all_label.append(1)
            # neg data
            c_id = [i for i in range(len(text_list)) if i != tidx and i != tidx + 1]
            other_idx = random.choice(c_id)
            other_text = text_list[other_idx]
            all_text1.append(text)
            all_text2.append(other_text)
            all_label.append(0)

        return all_text1, all_text2, all_label

    def build_task2_dataset(self, text_list: List):
        """
        利用文本分割和正负样本生成逻辑，为整个文本列表构建一个新的数据集，
        该数据集包含两个文本列（text1和text2）和一个标签列（label），
        并将这些数据保存到一个新的CSV文件中
        """
        if not self.output_task_file_path.exists():
            all_text1 = []
            all_text2 = []
            all_label = []
            for text in tqdm(text_list):
                # text split
                sp_text = self.__split_text(text)
                # split stop
                if len(sp_text) <= 2:
                    continue
                # build neg/pos pair data
                text1, text2, label = self.__build_neg_pos_data(sp_text)
                # data collect
                all_text1.extend(text1)
                all_text2.extend(text2)
                all_label.extend(label)
            df = pd.DataFrame({
                "text1": all_text1, 
                "text2": all_text2, 
                "label": all_label
            })
            df.to_csv(self.output_task_file_path, encoding="utf-8", index=False)

    def build_word_2_index(self, text_list: List):
        if not self.index_2_word_file_path.exists():
            # create index_2_word.txt file
            """
            在BERT论文中，这些特殊标记具有以下含义：
                - "[PAD]"：这是一个填充标记（Padding Token），用于将所有输入序列统一到相同的长度。在处理不等长的文本数据时，较短的序列会用[PAD]标记填充，以便能够批量处理。
                - "[unused1]"：这个标记在BERT的实现中并不使用，它是一个保留的标记，可能用于未来的扩展或者特定用途，但在当前的BERT模型中没有被赋予具体功能。
                - "[CLS]"：这是一个特殊的分类标记（Classification Token）。在BERT模型中，这个标记通常放在输入序列的开始位置。经过模型处理后，[CLS]标记的最终隐藏状态被用作整个输入序列的聚合表示，尤其在分类任务中。
                - "[SEP]"：这是一个分隔标记（Separator Token），用于分隔输入序列中的不同部分，比如句子对中的句子。在句子对任务中，[SEP]标记用来区分两个句子的边界。
                - "[MASK]"：这是一个掩码标记（Mask Token），用于BERT的预训练任务之一——Masked Language Model（MLM）。在MLM任务中，输入序列中的一些单词会被随机替换为[MASK]标记，模型的任务是预测这些被掩码的单词。
                - "[UNK]"：这是一个未知词标记（Unknown Token），用于表示词汇表中不存在的单词。当输入文本中的单词不在模型的词汇表内时，会用[UNK]标记来代替。
            """
            word_2_index = {
                "[PAD]": 0, "[unused1]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4, "[UNK]": 5,
            }
            for text in text_list:
                for w in text:
                    if w not in word_2_index:
                        word_2_index[w] = len(word_2_index)
            index_2_word = list(word_2_index)
            # write data
            with open(self.index_2_word_file_path, "w", encoding="utf-8") as file:
                file.write("\n".join(index_2_word))
            return word_2_index, index_2_word
        else:
            # read index_2_word.txt directly
            with open(self.index_2_word_file_path, "r", encoding="utf-8") as file:
                index_2_word = file.read().split("\n")
                word_2_index = {w: idx for idx, w in enumerate(index_2_word)}
                return word_2_index, index_2_word




# 测试代码 main 函数
def main():
    dataloader = DataProcessor()

    # read data
    all_text = dataloader.load_data()
    print(all_text[0])

    # build task2 data
    dataloader.build_task2_dataset(text_list=all_text)

    # build word_2_index
    word_2_index, index_2_word = dataloader.build_word_2_index(all_text)

if __name__ == "__main__":
    main()
