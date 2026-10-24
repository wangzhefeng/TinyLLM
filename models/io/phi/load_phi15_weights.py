# -*- coding: utf-8 -*-

# ***************************************************
# * File        : load_phi15_weights.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102421
# * Description : phi-1_5 官方权重下载与映射（HF PhiForCausalLM 命名）。
# *               合并 Wqkv 投影 (3d, d) 按行切片拆分为 W_query/W_key/W_value
# *               （HF safetensors 已是 nn.Linear (out, in) 布局，无需转置）；
# *               bias 同步按段拆分。
# * Link        : https://huggingface.co/microsoft/phi-1_5
# * Requirement : torch / safetensors / huggingface_hub
# ***************************************************

__all__ = ["download_phi15", "load_weights_into_phi", "model_with_phi15_weights"]

# python libraries
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from config.phi.model_cfgs import PHI15_CONFIG

PHI15_REPO_ID = "microsoft/phi-1_5"


def download_phi15(local_dir: str | None = None):
    """下载 phi-1_5 的 model.safetensors 与 tokenizer 文件。

    Returns:
        (weights_dict, local_dir)：safetensors 张量字典与本地目录 Path
    """
    local_dir = Path(local_dir or "./downloaded_models/phi_model/phi-1_5")
    local_dir.mkdir(parents=True, exist_ok=True)
    weights_file = local_dir / "model.safetensors"
    if not weights_file.exists():
        hf_hub_download(
            repo_id=PHI15_REPO_ID, filename="model.safetensors",
            local_dir=str(local_dir),
        )
    # tokenizer.json 必需；tokenizer_config 等尽力而为
    for fname in ("tokenizer.json", "tokenizer_config.json"):
        if not (local_dir / fname).exists():
            try:
                hf_hub_download(
                    repo_id=PHI15_REPO_ID, filename=fname,
                    local_dir=str(local_dir),
                )
            except Exception:
                if fname == "tokenizer.json":
                    raise
    weights = load_file(str(weights_file))
    return weights, local_dir


def load_weights_into_phi(model, params):
    """HF PhiForCausalLM 命名 -> models/phi 拆分式投影。

    仅支持 HF 格式（``model.layers.{i}.self_attn.qkv_proj.*``）；
    原仓 MixFormer 的 ``transformer.h.*`` 命名不在支持范围。
    """
    def assign(left, right, tensor_name="unknown"):
        if left.shape != right.shape:
            raise ValueError(
                f"Shape mismatch in '{tensor_name}': {left.shape} vs {right.shape}")
        with torch.no_grad():
            left.copy_(right)
        return left

    d = model.cfg.embed_dim
    assign(model.tok_embed.weight, params["model.embed_tokens.weight"],
           "model.embed_tokens.weight")
    for l in range(model.cfg.n_layers):
        block = model.blocks[l]
        p = f"model.layers.{l}"
        qkv_w = params[f"{p}.self_attn.qkv_proj.weight"]
        qkv_b = params[f"{p}.self_attn.qkv_proj.bias"]
        # 合并投影按行拆分：[0:d]=Q, [d:2d]=K, [2d:3d]=V
        assign(block.attn.W_query.weight, qkv_w[:d], f"{p}.qkv->W_query.weight")
        assign(block.attn.W_key.weight, qkv_w[d:2 * d], f"{p}.qkv->W_key.weight")
        assign(block.attn.W_value.weight, qkv_w[2 * d:], f"{p}.qkv->W_value.weight")
        assign(block.attn.W_query.bias, qkv_b[:d], f"{p}.qkv->W_query.bias")
        assign(block.attn.W_key.bias, qkv_b[d:2 * d], f"{p}.qkv->W_key.bias")
        assign(block.attn.W_value.bias, qkv_b[2 * d:], f"{p}.qkv->W_value.bias")
        assign(block.attn.out_proj.weight,
               params[f"{p}.self_attn.out_proj.weight"], f"{p}.out_proj.weight")
        assign(block.attn.out_proj.bias,
               params[f"{p}.self_attn.out_proj.bias"], f"{p}.out_proj.bias")
        assign(block.input_layernorm.scale,
               params[f"{p}.input_layernorm.weight"], f"{p}.input_layernorm.weight")
        assign(block.input_layernorm.shift,
               params[f"{p}.input_layernorm.bias"], f"{p}.input_layernorm.bias")
        assign(block.post_attention_layernorm.scale,
               params[f"{p}.post_attention_layernorm.weight"],
               f"{p}.post_attention_layernorm.weight")
        assign(block.post_attention_layernorm.shift,
               params[f"{p}.post_attention_layernorm.bias"],
               f"{p}.post_attention_layernorm.bias")
        assign(block.ff.fc1.weight, params[f"{p}.mlp.fc1.weight"], f"{p}.mlp.fc1.weight")
        assign(block.ff.fc1.bias, params[f"{p}.mlp.fc1.bias"], f"{p}.mlp.fc1.bias")
        assign(block.ff.fc2.weight, params[f"{p}.mlp.fc2.weight"], f"{p}.mlp.fc2.weight")
        assign(block.ff.fc2.bias, params[f"{p}.mlp.fc2.bias"], f"{p}.mlp.fc2.bias")
    assign(model.final_norm.scale, params["model.norm.weight"], "model.norm.weight")
    assign(model.final_norm.shift, params["model.norm.bias"], "model.norm.bias")
    if "lm_head.weight" in params:
        assign(model.out_head.weight, params["lm_head.weight"], "lm_head.weight")
    else:
        model.out_head.weight = model.tok_embed.weight
        print("Model uses weight tying")
    return model


def model_with_phi15_weights(model_path: str | None = None, device: str = "cpu"):
    """构建 phi-1.5 模型并加载官方权重，返回 (model, config)。"""
    from models.phi.phi15 import Model
    from utils.args_tools import DotDict

    config = DotDict(PHI15_CONFIG)
    model = Model(config)
    weights, _ = download_phi15(model_path)
    load_weights_into_phi(model, weights)
    del weights
    model.to(device)
    model.eval()
    return model, config


# 测试代码 main 函数
def main():
    from utils.log_util import logger

    model, config = model_with_phi15_weights()
    logger.info(f"phi-1.5 loaded: {sum(p.numel() for p in model.parameters()):,} params")

if __name__ == "__main__":
    main()
