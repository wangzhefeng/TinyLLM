# -*- coding: utf-8 -*-

# ***************************************************
# * File        : plot_losses.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-23
# * Version     : 0.1.022300
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def plot_values_classifier(train_epochs, examples_seen, train_values, val_values, label: str = "loss", results_path: str = None):
    # epochs tensor
    epochs_tensor = torch.linspace(0, train_epochs, len(train_values))
    
    # plot training and validation loss against epochs
    fig, ax1 = plt.subplots(figsize = (5, 3))
    ax1.plot(epochs_tensor, train_values, label = f"Training {label}")
    ax1.plot(epochs_tensor, val_values, linestyle = "-.", label = f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()
    # only show integer labels on x-axis
    ax1.xaxis.set_major_locator(MaxNLocator(integer = True))
    
    # create a second x-axis for examples seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_values))
    ax2.plot(examples_seen_tensor, train_values, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Examples seen")
    
    # adjust layout to make room
    fig.tight_layout()
    # save fig
    plt.savefig(Path(results_path).joinpath(f"{label}-plot.pdf"))
    # show fig on notebook
    plt.show()


def plot_losses(train_epochs, tokens_seen, train_losses, val_losses, label: str = "loss", results_path: str = None):
    # epochs seen
    epochs_seen = torch.linspace(0, train_epochs, len(train_losses))
    
    # plot training and validation loss against epochs
    fig, ax1 = plt.subplots(figsize = (5, 3))
    ax1.plot(epochs_seen, train_losses, label = f"Training {label}")
    ax1.plot(epochs_seen, val_losses, linestyle = "-.", label = f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend(loc = "upper right")
    # only show integer labels on x-axis
    ax1.xaxis.set_major_locator(MaxNLocator(integer = True))
    
    # create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha = 0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")
    
    # adjust layout to make room
    fig.tight_layout()
    # save fig
    plt.savefig(Path(results_path).joinpath(f"{label}_plot.pdf"))
    # show fig
    plt.show()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
