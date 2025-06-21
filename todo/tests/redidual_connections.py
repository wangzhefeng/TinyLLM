# -*- coding: utf-8 -*-

# ***************************************************
# * File        : redidual_connections.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-27
# * Version     : 1.0.012711
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)


import torch
import torch.nn as nn

from layers.activation import GELU

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class DeepNeuralNetwork(nn.Module):
    
    def __init__(self, layer_sizes, use_shortcut):
        super(DeepNeuralNetwork, self).__init__()

        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(layer_sizes[0], layer_sizes[1]), 
                GELU()
            ),
            nn.Sequential(
                nn.Linear(layer_sizes[1], layer_sizes[2]),
                GELU(),
            ),
            nn.Sequential(
                nn.Linear(layer_sizes[2], layer_sizes[3]),
                GELU(),
            ),
            nn.Sequential(
                nn.Linear(layer_sizes[3], layer_sizes[4]),
                GELU(),
            ),
            nn.Sequential(
                nn.Linear(layer_sizes[4], layer_sizes[5]),
                GELU(),
            )
        ])

    def forward(self, x):
        for layer in self.layers:
            # compute the output of the current layer
            layer_output = layer(x)
            # check if shortcut can be applied
            if self.use_shortcut and x.shape == layer_output.shape:
                x = layer_output + x
            else:
                x = layer_output
        
        return x


def print_gradients(model, x):
    # forward class
    output = model(x)
    target = torch.tensor([[0.0]])
    # calculate loss based on how close the target and output are
    loss = nn.MSELoss()
    loss = loss(output, target)
    # backward pass to calculate the gradients
    loss.backward()
    
    for name, param in model.named_parameters():
        if "weight" in name:
            # print the mean absolute gradient of the weights
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")




# 测试代码 main 函数
def main():
    # ------------------------------
    # gradient values without shortcut connections
    # ------------------------------
    layer_sizes = [3, 3, 3, 3, 3, 1]
    sample_input = torch.tensor([[1.0, 0.0, -1.0]])
    torch.manual_seed(123)
    model_without_shortcut = DeepNeuralNetwork(layer_sizes, use_shortcut=False)
    print_gradients(model_without_shortcut, sample_input)
    print()
    # ------------------------------
    # gradient values with shortcut connections
    # ------------------------------
    torch.manual_seed(123)
    model_with_shortcut = DeepNeuralNetwork(layer_sizes, use_shortcut=True)
    print_gradients(model_with_shortcut, sample_input)

if __name__ == "__main__":
    main()
