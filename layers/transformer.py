# -*- coding: utf-8 -*-

# ***************************************************
# * File        : transformer.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-24
# * Version     : 1.0.082420
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

import torch.nn as nn

from layers.transformer_encdec import Encoder, Decoder

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Transformer(nn.Module):
    """
    Encoder-Decoder

    A standard Encoder-Decoder architecture. Base for this and many other models.
    """
    
    def __init__(self, woe: str, wpe: str, 
                 src_vocab_size: int, tgt_vocab_size: int, 
                 d_model: int=512, n_heads: int=8, d_ff: int=2048, 
                 num_layers: int=6, dropout: float=0.1):
        super().__init__()

        self.encoder = Encoder(
            woe, wpe,
            src_vocab_size, 
            d_model, 
            n_heads, 
            d_ff, 
            num_layers, 
            dropout
        )
        self.decoder = Decoder(
            woe, 
            wpe,
            tgt_vocab_size, 
            d_model, 
            n_heads, 
            d_ff, 
            num_layers, 
            dropout
        )
        self.linear = nn.Linear(d_model, tgt_vocab_size)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # encoder
        enc_output = self.encoder(src, src_mask)
        # decoder
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        # final linear projection
        output = self.linear(dec_output)

        return output
    
    def encode(self, src, src_mask=None):
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        return self.decoder(tgt, enc_output, src_mask, tgt_mask)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
