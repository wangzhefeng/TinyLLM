# -*- coding: utf-8 -*-

# ***************************************************
# * File        : masking.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-13
# * Version     : 1.0.071303
# * Description : description
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
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import torch

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def create_padding_mask(seq, pad_idx=0):
    """
    Padding mask: Ensures the model ignores padding tokens
    create a mask for padding tokens: 1 for non-pad, 0 for pad

    Args:
        seq (_type_): _description_
        pad_idx (int, optional): _description_. Defaults to 0.
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def create_look_ahead_mask(size):
    """
    Look-ahead mask: Prevents the decoder from seeing future tokens during training
    create a triangular mask to hide future tokens

    Args:
        size (_type_): _description_
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()

    return ~mask


def subsequent_mask(size):
    """
    Mask out subsequent positions
    """
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(
        torch.ones(attn_shape), 
        diagonal=1
    ).type(torch.uint8)
    print(subsequent_mask)

    return subsequent_mask == 0


# TODO
def example_mask():
    import altair as alt
    LS_data = pd.concat([
        pd.DataFrame({
            "Subsequent Mask": subsequent_mask(20)[0][x, y].flatten(),
            "Window": y,
            "Masking": x
        })
        for y in range(20)
        for x in range(20)
    ])
    print(LS_data)
    chart = alt.Chart(LS_data) \
        .mark_rect() \
        .properties(height=250, width=250) \
        .encode(
            alt.X("Window:O"),
            alt.Y("Masking:O"),
            alt.Color("Subsequent Mask:Q", scale=alt.Scale(scheme="viridis")),
        ) \
        .interactive()

    return chart




# 测试代码 main 函数
def main():
    chart = example_mask()

if __name__ == "__main__":
    main()
