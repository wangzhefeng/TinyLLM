# -*- coding: utf-8 -*-

# ***************************************************
# * File        : FixPE.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-09
# * Version     : 1.0.070922
# * Description : Fixed Position Encoding
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import math
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class PositionalEncoding(nn.Module):
    """
    位置编码模块
    
    在输入(input -> Embedding)上加入了位置编码
    """
    
    def __init__(self, d_model: int=512, max_len: int=5000):
        super().__init__()

        # create positional encoding matrix
        pe = torch.zeros(max_len, d_model).float()
        # TODO pe.requires_grad = False
        # pos
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        # 1 / 10000^{2i/d_model}
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # apply sine to even indices: pe(pos, 2i) = sin(pos / 10000^{2i/d_model})
        pe[:, 0::2] = torch.sin(position * div_term)
        # apply cosine to odd indices: pe(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # add batch dimension
        pe = pe.unsqueeze(0)
        
        # register the positional encoding as a buffer (not a parameter)
        self.register_buffer("pe", pe)
    
    def forward(self, x):
        """
        add positional encoding to the input tensor
        """
        # x.shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]

        return x


class PositionalEncoding(nn.Module):
    """
    位置编码模块
    
    在输入(input -> Embedding)上加入了位置编码
    """

    def __init__(self, config):
        super().__init__()

        # Dropout 层
        self.dropout = nn.Dropout(p = config.dropout)
        
        # Position Embeding 层
        pe = torch.zeros(config.block_size, config.n_embd).float()
        pe.requires_grad = False
        # pos
        position = torch.arange(0, config.block_size, dtype=torch.float32).unsqueeze(1)
        # 1 / 10000^{2i/d_model}
        div_term = torch.exp(torch.arange(0, config.n_embd, 2).float() * -(math.log(10000.0) / config.n_embd))
        
        # apply sine to even indices: pe(pos, 2i) = sin(pos / 10000^{2i/d_model})
        pe[:, 0::2] = torch.sin(position * div_term)
        # apply cosine to odd indices: pe(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
        pe[:, 1::2] = torch.cos(position * div_term)

        # add batch dimension
        pe = pe.unsqueeze(0)

        # register the positional encoding as a buffer (not a parameter)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        add positional encoding to the input tensor
        """
        # x.shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        # dropout
        x = self.dropout(x)

        return x


def example_positional():
    import pandas as pd
    import altair as alt

    pe = PositionalEncoding(20, 0)
    y = pe.forward(torch.zeros(1, 100, 20))

    data = pd.concat(
        [
            pd.DataFrame(
                {
                    "embedding": y[0, :, dim],
                    "dimension": dim,
                    "position": list(range(100)),
                }
            )
            for dim in [4, 5, 6, 7]
        ]
    )

    return (
        alt.Chart(data)
        .mark_line()
        .properties(width=800)
        .encode(x="position", y="embedding", color="dimension:N")
        .interactive()
    )




# 测试代码 main 函数
def main():
    import numpy as np
    # input
    input_tensor = torch.from_numpy(np.array(
        [[[1,  2,  3,  4,  5,  6],
          [5,  6,  7,  8,  9,  10],
          [9,  10, 11, 12, 13, 14],
          [13, 14, 15, 16, 17, 18]]]
    ))
    # print(input_tensor)
    print(input_tensor.size())

    # positional encoding
    pe = PositionalEncoding(d_model=4, max_len=6)

    # forward
    # output = pe(input_tensor)

if __name__ == "__main__":
    main()
