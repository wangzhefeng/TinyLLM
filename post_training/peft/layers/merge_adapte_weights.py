# -*- coding: utf-8 -*-

# ***************************************************
# * File        : merge_adapte_weights.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-21
# * Version     : 1.1.102121
# * Description : LoRA/适配器权重合并工具（兼容转发入口）。
# *               规范实现自 2026-10-20 起在 layers/lora.py；
# *               本文件保留历史导入路径 `post_training.peft.layers.merge_adapte_weights`。
# *               注意：文件名中的 "adapte" 为历史拼写（应为 adapter），
# *               为兼容既有 import 不改名；新代码请直接使用 layers.lora。
# * Link        : LLMs-from-scratch 附录 E
# *               https://github.com/rasbt/LLMs-from-scratch/tree/main/appendix-E
# * Requirement : torch
# ***************************************************

from layers.lora import (
    freeze_linear_layers,
    merge_lora_weights,
    replace_linear_with_lora,
)

__all__ = [
    "freeze_linear_layers",
    "merge_lora_weights",
    "replace_linear_with_lora",
]




# 测试代码 main 函数
def main():
    # 用法示例见 tests/unit/test_lora.py
    pass

if __name__ == "__main__":
    main()
