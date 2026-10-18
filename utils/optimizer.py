# -*- coding: utf-8 -*-

# ***************************************************
# * File        : optimizer.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-09-13
# * Version     : 1.0.091321
# * Description : optimizer factory
# * Link        : link
# * Requirement : galore-torch (optional, only for galore_* types)
# ***************************************************

__all__ = [
    "select_optimizer",
]

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

import torch

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def select_optimizer(model, learning_rate: float, weight_decay: float, optimizer_type: str="adamw"):
    """
    optimizer

    Args:
        model: nn.Module
        learning_rate: learning rate
        weight_decay: weight decay
        optimizer_type: "adamw" | "galore_adamw" | "galore_adamw8bit" | "galore_adafactor"

    Returns:
        torch.optim.Optimizer
    """
    if optimizer_type == "adamw":
        kwargs = {
            "lr": learning_rate,
            "weight_decay": weight_decay,
        }
        # fused AdamW 只在 CUDA 上可用（MPS / CPU 会报错）
        if torch.cuda.is_available():
            kwargs["fused"] = True
        optimizer = torch.optim.AdamW(model.parameters(), **kwargs)
    elif optimizer_type in ("galore_adamw", "galore_adamw8bit", "galore_adafactor"):
        try:
            from galore_torch import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor
        except ImportError as exc:
            raise ImportError(
                f"optimizer_type={optimizer_type!r} requires `galore-torch`, "
                "install it with: pip install galore-torch"
            ) from exc
        optimizers = {
            "galore_adamw": GaLoreAdamW,
            "galore_adamw8bit": GaLoreAdamW8bit,
            "galore_adafactor": GaLoreAdafactor,
        }
        optimizer = optimizers[optimizer_type](
            model.parameters(),
            lr = learning_rate,
            weight_decay = weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer_type: {optimizer_type!r}")

    return optimizer




# 测试代码 main 函数
def main():
    model = torch.nn.Linear(4, 2)
    optimizer = select_optimizer(model, learning_rate=1e-3, weight_decay=0.1)
    print(optimizer)

if __name__ == "__main__":
    main()
