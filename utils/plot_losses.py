# -*- coding: utf-8 -*-

# ***************************************************
# * File        : plot_losses.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-23
# * Version     : 0.1.022300
# * Description : plot loss curves
# * Link        : link
# * Requirement : matplotlib (optional)
# ***************************************************

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def _import_pyplot():
    """
    matplotlib 是可选依赖：缺失时给出提示并返回 None，
    避免仅仅因为没有安装 matplotlib 就让训练主流程导入失败。
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        logger.warning(
            "matplotlib is not installed, skip plotting loss curves. "
            "Install it with: uv add matplotlib"
        )
        return None, None

    return plt, MaxNLocator


def _prepare_results_dir(results_path):
    if results_path is not None:
        Path(results_path).mkdir(parents=True, exist_ok=True)

    return Path(results_path) if results_path is not None else Path(".")


def plot_values_classifier(train_epochs, examples_seen, train_values, val_values, label: str="loss", results_path: str=None):
    plt, MaxNLocator = _import_pyplot()
    if plt is None:
        return None
    results_path = _prepare_results_dir(results_path)
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
    # grid
    plt.grid(True)
    # save fig
    plt.savefig(results_path.joinpath(f"{label}-plot.pdf"))
    plt.close(fig)


def plot_losses_llm(train_epochs, tokens_seen, train_losses, valid_losses, label: str="loss", results_path: str=None):
    plt, MaxNLocator = _import_pyplot()
    if plt is None:
        return None
    results_path = _prepare_results_dir(results_path)
    # epochs seen
    epochs_seen = torch.linspace(0, train_epochs, len(train_losses))
    # plot training and validation loss against epochs
    fig, ax1 = plt.subplots(figsize = (5, 3))
    ax1.plot(epochs_seen, train_losses, label = f"Training {label}")
    ax1.plot(epochs_seen, valid_losses, label = f"Validation {label}", linestyle = "-.")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend(loc="upper right")
    # only show integer labels on x-axis
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    # create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha = 0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")
    # adjust layout to make room
    fig.tight_layout()
    # grid
    plt.grid(True)
    # save fig
    plt.savefig(results_path.joinpath(f"{label}_plot.pdf"))
    plt.close(fig)


def plot_losses(train_epochs, train_losses, valid_losses, label: str="loss", results_path: str=None):
    plt, MaxNLocator = _import_pyplot()
    if plt is None:
        return None
    results_path = _prepare_results_dir(results_path)
    # epochs seen
    epochs_seen = torch.linspace(0, train_epochs, len(train_losses))
    # plot training and validation loss against epochs
    fig, ax1 = plt.subplots(figsize = (5, 3))
    ax1.plot(epochs_seen, train_losses, label = f"Training {label}")
    ax1.plot(epochs_seen, valid_losses, label = f"Validation {label}", linestyle = "-.")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend(loc="upper right")
    # only show integer labels on x-axis
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    # adjust layout to make room
    fig.tight_layout()
    # grid
    plt.grid(True)
    # save fig
    plt.savefig(results_path.joinpath(f"{label}_plot.pdf"))
    plt.close(fig)




# 测试代码 main 函数
def main():
    plot_losses(
        train_epochs=10,
        train_losses=[1.0, 0.8, 0.6, 0.5],
        valid_losses=[1.1, 0.9, 0.7, 0.6],
        label="loss",
        results_path="./saved_results/test_results/",
    )

if __name__ == "__main__":
    main()
