# -*- coding: utf-8 -*-

# ***************************************************
# * File        : loader.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-28
# * Version     : 1.0.072816
# * Description : 权重加载工具。
# *               1. init_random_weights(): 随机初始化(教学演示用)
# *               2. load_state_dict(): 识别参数上的 weight_loader 钩子，
# *                  按 TP 语义加载外部 state_dict
# *               3. load_model(): 从 HF 格式的本地目录(*.safetensors)加载
# *                  真实权重(对齐上游 nanovllm/utils/loader.py)
# * Link        : https://github.com/GeeeekExplorer/nano-vllm
# * Requirement : torch >= 2.0, safetensors
# ***************************************************

__all__ = [
    "init_random_weights",
    "load_state_dict",
    "load_model",
]

# python libraries
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

from ..layers.parallel_linear import (
    LinearBase,
    QKVParallelLinear,
    MergedColumnParallelLinear,
)

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def _param_seed(seed: int, full_name: str) -> int:
    """
    由 (seed, 参数名) 派生独立种子：保证同一参数在任何 rank / 任何 TP 切分下
    生成的"全局权重"完全一致(多进程 TP 一致性所必需)。
    """
    import zlib
    return (seed * 1_000_003 + zlib.crc32(full_name.encode())) % (2**63)


def _full_shape(module: nn.Module, param: nn.Parameter) -> tuple[int, ...]:
    """
    根据模块类型推导该参数的"未分片全局形状"
    """
    if isinstance(module, LinearBase):
        return (module.output_size, module.input_size)
    if hasattr(module, "num_embed"):  # VocabParallelEmbedding / ParallelLMHead
        return (module.num_embed, param.shape[-1])
    return tuple(param.shape)


def _load_sharded(module: nn.Module, param: nn.Parameter, full_weight: torch.Tensor):
    """
    把全局权重按 TP 语义写入(可能已分片的)参数：
    有 weight_loader 钩子且形状不同 -> 走钩子分片；否则直接 copy_
    """
    weight_loader = getattr(param, "weight_loader", None)
    if weight_loader is None or full_weight.shape == param.shape:
        param.data.copy_(full_weight)
        return
    if isinstance(module, QKVParallelLinear):
        # 全局布局 [q | k | v]，逐 shard 交给钩子定位
        q_size = module.num_heads * module.tp_size * module.head_size
        kv_size = module.num_kv_heads * module.tp_size * module.head_size
        weight_loader(param, full_weight[:q_size], "q")
        weight_loader(param, full_weight[q_size: q_size + kv_size], "k")
        weight_loader(param, full_weight[q_size + kv_size:], "v")
    elif isinstance(module, MergedColumnParallelLinear):
        offset = 0
        for shard_id, size in enumerate(module.output_sizes):
            weight_loader(param, full_weight[offset: offset + size], shard_id)
            offset += size
    else:
        weight_loader(param, full_weight)


@torch.no_grad()
def init_random_weights(model: nn.Module, seed: int = 42, std: float = 0.02):
    """
    随机初始化模型权重(不加载任何外部权重文件)

    - 2 维及以上参数(embedding / linear weight): N(0, std)，按参数名播种，
      先生成"全局权重"再经 weight_loader 分片，保证 tp=1 与 tp=N 结果一致
    - bias: 0
    - norm 层 weight(构造时已是全 1): 保持不变
    - tie_word_embeddings 共享的参数只初始化一次
    """
    initialized = set()
    for mod_name, module in model.named_modules():
        for pname, param in module.named_parameters(recurse=False):
            if id(param) in initialized:
                continue
            initialized.add(id(param))
            if pname == "bias":
                param.zero_()
                continue
            if param.dim() == 1:
                continue  # RMSNorm.weight 等一维权重构造时已初始化为 1
            full_name = f"{mod_name}.{pname}"
            generator = torch.Generator(device="cpu").manual_seed(_param_seed(seed, full_name))
            full_weight = torch.empty(_full_shape(module, param)).normal_(0.0, std, generator=generator)
            _load_sharded(module, param, full_weight)
    return model


@torch.no_grad()
def load_state_dict(model: nn.Module, state_dict: dict, packed_modules_mapping: dict | None = None):
    """
    按 TP 语义加载 state_dict：若参数上挂载了 weight_loader 钩子则走钩子(支持分片/打包)，
    否则直接 copy_。packed_modules_mapping 用于把拆分权重(如 q_proj)映射到打包参数(qkv_proj)。
    """
    packed_modules_mapping = packed_modules_mapping or getattr(model, "packed_modules_mapping", {})
    for name, loaded_weight in state_dict.items():
        # 拆分权重映射到打包参数，如 self_attn.q_proj.weight -> self_attn.qkv_proj.weight ("q")
        shard_id = None
        for key, (packed_key, shard) in packed_modules_mapping.items():
            if f".{key}." in name or name.endswith(f".{key}"):
                name = name.replace(key, packed_key)
                shard_id = shard
                break
        # 逐层解析属性找到目标参数
        module, attr = model, name
        parts = name.split(".")
        for part in parts[:-1]:
            module = getattr(module, part) if not part.isdigit() else module[int(part)]
        param = getattr(module, parts[-1])
        weight_loader = getattr(param, "weight_loader", None)
        if weight_loader is not None:
            if shard_id is not None:
                weight_loader(param, loaded_weight, shard_id)
            else:
                weight_loader(param, loaded_weight)
        else:
            param.data.copy_(loaded_weight)
    return model


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


@torch.no_grad()
def load_model(model: nn.Module, path: str):
    """
    从 HF 格式的本地目录加载真实权重(对齐上游 nanovllm/utils/loader.py)：
    遍历目录下所有 *.safetensors 文件，拆分权重(q_proj/gate_proj 等)经
    packed_modules_mapping 映射到打包参数(qkv_proj/gate_up_proj)，
    并走参数上的 weight_loader 钩子完成 TP 分片/打包写入。

    Args:
        model: Qwen3ForCausalLM(需带 packed_modules_mapping)
        path: 含有 *.safetensors 的模型目录(如 ~/huggingface/Qwen3-0.6B/)
    """
    import os
    from glob import glob
    from safetensors import safe_open

    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    files = sorted(glob(os.path.join(path, "*.safetensors")))
    assert files, f"未在 {path} 找到 *.safetensors 权重文件"
    for file in files:
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
    return model




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
