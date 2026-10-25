# -*- coding: utf-8 -*-

# ***************************************************
# * File        : hf_export.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102421
# * Description : 本地 Qwen3 权重 -> HF Transformers 兼容目录导出（ch08）。
# *               1. export_to_hf：本地 state_dict 键名反向映射到 HF Qwen3
# *                  命名（load_weights_into_qwen 的逆映射），safetensors 落盘
# *                  model.safetensors，并生成 HF 格式 config.json 与
# *                  generation_config.json
# *               2. TinyLLMForCausalLM：wrapper 方案，PreTrainedModel 包住
# *                  本地 Model 的 forward（transformers 懒加载，缺失时模块
# *                  仍可导入）
# * Link        : https://github.com/rasbt/reasoning-from-scratch/tree/main/ch08/06_use_via_huggingface
# * Requirement : torch / safetensors；transformers 仅 TinyLLMForCausalLM 需要
# ***************************************************

__all__ = [
    "export_to_hf",
    "local_to_hf_state_dict",
    "TinyLLMConfig",
    "TinyLLMForCausalLM",
]

# python libraries
import json
import re
from pathlib import Path

import torch

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]

# 本地命名 -> HF Qwen3 命名（models/io/qwen3/load_weights.py 的反向映射）
_KEY_REPLACEMENTS = [
    ("tok_embed.weight", "model.embed_tokens.weight"),
    ("final_norm.scale", "model.norm.weight"),
    ("out_head.weight", "lm_head.weight"),
    (".attn.q_norm.scale", ".self_attn.q_norm.weight"),
    (".attn.k_norm.scale", ".self_attn.k_norm.weight"),
    (".attn.W_query.weight", ".self_attn.q_proj.weight"),
    (".attn.W_key.weight", ".self_attn.k_proj.weight"),
    (".attn.W_value.weight", ".self_attn.v_proj.weight"),
    (".attn.out_proj.weight", ".self_attn.o_proj.weight"),
    (".ff.gate.weight", ".mlp.gate.weight"),
    (".ff.fc1.weight", ".mlp.gate_proj.weight"),
    (".ff.fc2.weight", ".mlp.up_proj.weight"),
    (".ff.fc3.weight", ".mlp.down_proj.weight"),
    (".norm1.scale", ".input_layernorm.weight"),
    (".norm2.scale", ".post_attention_layernorm.weight"),
]
_EXPERT_RE = re.compile(r"^trf_blocks\.(\d+)\.ff\.fc([123])\.(\d+)\.weight$")
_EXPERT_PROJ = {"1": "gate_proj", "2": "up_proj", "3": "down_proj"}


def local_to_hf_state_dict(state_dict):
    """本地 Qwen3 键名反向映射到 HF Qwen3 命名（dense + MoE experts）。

    无匹配规则时抛 KeyError，避免静默导出错误命名。
    """
    hf_state_dict = {}
    for key, tensor in state_dict.items():
        expert = _EXPERT_RE.match(key)
        if expert is not None:
            layer, fc, expert_idx = expert.groups()
            new_key = (
                f"model.layers.{layer}.mlp.experts.{expert_idx}"
                f".{_EXPERT_PROJ[fc]}.weight"
            )
        else:
            new_key = key
            for old, new in _KEY_REPLACEMENTS:
                if old in new_key:
                    new_key = new_key.replace(old, new)
            new_key = new_key.replace("trf_blocks.", "model.layers.")
            if new_key == key:
                raise KeyError(f"无法映射本地键名到 HF 命名: {key}")
        hf_state_dict[new_key] = tensor
    return hf_state_dict


def _cfg_to_dict(cfg):
    """ModelConfig / dict / SimpleNamespace -> 纯 dict。"""
    if hasattr(cfg, "to_dict"):
        return dict(cfg.to_dict())
    if isinstance(cfg, dict):
        return dict(cfg)
    return dict(vars(cfg))


def _dtype_to_name(dtype_value):
    if isinstance(dtype_value, torch.dtype):
        return str(dtype_value).removeprefix("torch.")
    return dtype_value


def _tokenizer_special_ids(tokenizer_dir):
    """从本地 tokenizer.json 解析 pad/eos id（解析失败返回空 dict）。

    特殊 token 可能在 model.vocab，也可能在 added_tokens（HF fast 格式）。
    """
    tokenizer_file = Path(tokenizer_dir) / "tokenizer.json"
    if not tokenizer_file.exists():
        return {}
    try:
        data = json.loads(tokenizer_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    vocab = data.get("model", {}).get("vocab", {})
    vocab.update({
        token["content"]: token["id"]
        for token in data.get("added_tokens", [])
        if "content" in token and "id" in token
    })
    ids = {}
    if "<|endoftext|>" in vocab:
        ids["pad_token_id"] = vocab["<|endoftext|>"]
    if "<|im_end|>" in vocab:
        ids["eos_token_id"] = vocab["<|im_end|>"]
    return ids


def export_to_hf(model, out_dir, cfg, tokenizer_dir=None):
    """本地 Qwen3 模型 -> HF Transformers 兼容目录（ch08 export 方案）。

    落盘 ``model.safetensors``（键名已映射到 HF Qwen3 命名）、
    ``config.json``（Qwen3ForCausalLM 字段，从 cfg 推导）与
    ``generation_config.json``（eos/pad 优先取 tokenizer_dir）。

    Args:
        model: 本地 Qwen3 Model（models.qwen3.qwen3）
        out_dir: 输出目录（不存在则创建）
        cfg: 模型配置（ModelConfig / dict / SimpleNamespace）
        tokenizer_dir: 可选，含 tokenizer.json 的目录（推导 eos/pad id）

    Returns:
        输出目录 Path
    """
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("未安装 safetensors，请先 pip install safetensors") from exc

    cfg_dict = _cfg_to_dict(cfg)
    if cfg_dict.get("num_experts", 0) > 0:
        raise NotImplementedError("MoE 模型导出未实现（仅支持 dense Qwen3）")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hf_state_dict = {
        key: value.detach().cpu().contiguous().clone()
        for key, value in local_to_hf_state_dict(model.state_dict()).items()
    }
    save_file(hf_state_dict, str(out_dir / "model.safetensors"))

    special_ids = _tokenizer_special_ids(tokenizer_dir) if tokenizer_dir else {}
    config_json = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "vocab_size": cfg_dict["vocab_size"],
        "hidden_size": cfg_dict["embed_dim"],
        "num_hidden_layers": cfg_dict["n_layers"],
        "num_attention_heads": cfg_dict["n_heads"],
        "num_key_value_heads": cfg_dict["n_kv_groups"],
        "head_dim": cfg_dict["head_dim"],
        "intermediate_size": cfg_dict["d_ff"],
        "max_position_embeddings": cfg_dict["context_length"],
        "rope_theta": cfg_dict["rope_base"],
        "rms_norm_eps": 1e-6,
        "qk_norm": bool(cfg_dict.get("qk_norm", True)),
        "tie_word_embeddings": False,
        "attention_bias": False,
        "use_cache": True,
        "torch_dtype": _dtype_to_name(cfg_dict.get("dtype", torch.float32)),
        **special_ids,
    }
    (out_dir / "config.json").write_text(
        json.dumps(config_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    generation_config_json = {
        key: value for key, value in {
            "eos_token_id": special_ids.get("eos_token_id"),
            "pad_token_id": special_ids.get("pad_token_id"),
            "do_sample": False,
            "use_cache": False,
        }.items() if value is not None
    }
    (out_dir / "generation_config.json").write_text(
        json.dumps(generation_config_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Exported HF model folder to {out_dir}")
    return out_dir


# ------------------------------
# wrapper 方案（transformers 懒加载，模块导入零副作用）
# ------------------------------
_WRAPPER_CLASSES = None


def _cfg_jsonable(model_cfg):
    model_cfg = dict(model_cfg)
    if "dtype" in model_cfg:
        model_cfg["dtype"] = _dtype_to_name(model_cfg["dtype"])
    return model_cfg


def _cfg_from_jsonable(model_cfg):
    model_cfg = dict(model_cfg)
    if isinstance(model_cfg.get("dtype"), str):
        model_cfg["dtype"] = getattr(torch, model_cfg["dtype"])
    return model_cfg


def _load_wrapper_classes():
    """惰性构造 PreTrainedModel 包装器（首次访问 TinyLLM* 时 import transformers）。"""
    global _WRAPPER_CLASSES
    if _WRAPPER_CLASSES is not None:
        return _WRAPPER_CLASSES
    try:
        from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
        from transformers.modeling_outputs import CausalLMOutputWithPast
    except ImportError as exc:
        raise ImportError(
            "TinyLLMForCausalLM 需要 transformers，请先 pip install transformers"
        ) from exc

    class TinyLLMConfig(PretrainedConfig):
        """把本地 Qwen3 cfg 序列化进 PretrainedConfig（model_cfg 字段）。"""

        model_type = "tinyllm_local_wrapper"

        def __init__(self, model_cfg=None, **kwargs):
            model_cfg = _cfg_jsonable(model_cfg or {})
            self.model_cfg = model_cfg
            self.vocab_size = model_cfg.get("vocab_size")
            self.hidden_size = model_cfg.get("embed_dim")
            self.intermediate_size = model_cfg.get("d_ff")
            self.num_hidden_layers = model_cfg.get("n_layers")
            self.num_attention_heads = model_cfg.get("n_heads")
            self.num_key_value_heads = model_cfg.get("n_kv_groups")
            self.head_dim = model_cfg.get("head_dim")
            self.max_position_embeddings = model_cfg.get("context_length")
            kwargs.pop("tie_word_embeddings", None)
            super().__init__(
                torch_dtype=model_cfg.get("dtype"),
                use_cache=False,
                tie_word_embeddings=False,
                **kwargs,
            )

    class TinyLLMForCausalLM(PreTrainedModel, GenerationMixin):
        """包住本地 Model 的 forward（input_ids -> logits），不复制实现。"""

        config_class = TinyLLMConfig
        main_input_name = "input_ids"

        def __init__(self, config):
            super().__init__(config)
            from types import SimpleNamespace
            from models.qwen3.qwen3 import Model

            self.model = Model(SimpleNamespace(**_cfg_from_jsonable(config.model_cfg)))
            self.generation_config.use_cache = False
            self.generation_config.pad_token_id = getattr(config, "pad_token_id", None)
            self.generation_config.eos_token_id = getattr(config, "eos_token_id", None)

        def get_input_embeddings(self):
            return self.model.tok_embed

        def set_input_embeddings(self, value):
            self.model.tok_embed = value

        def get_output_embeddings(self):
            return self.model.out_head

        def set_output_embeddings(self, value):
            self.model.out_head = value

        def forward(self, input_ids=None, attention_mask=None, labels=None,
                    use_cache=None, return_dict=None, **kwargs):
            del attention_mask, use_cache, kwargs
            if input_ids is None:
                raise ValueError("input_ids must be provided.")
            logits = self.model(input_ids)
            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
            if return_dict is False:
                outputs = (logits,)
                return ((loss,) + outputs) if loss is not None else outputs
            return CausalLMOutputWithPast(loss=loss, logits=logits)

    _WRAPPER_CLASSES = {
        "TinyLLMConfig": TinyLLMConfig,
        "TinyLLMForCausalLM": TinyLLMForCausalLM,
    }
    return _WRAPPER_CLASSES


def __getattr__(name):
    if name in ("TinyLLMConfig", "TinyLLMForCausalLM"):
        return _load_wrapper_classes()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
