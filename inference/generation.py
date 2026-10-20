# -*- coding: utf-8 -*-

# ***************************************************
# * File        : generation.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-20
# * Version     : 1.0.102020
# * Description : 统一文本生成入口 generate_text()。
# *               按模型族分派到 layers/inference.py 的教材实现
# *               （gpt2/llama → generate；qwen3 → generate_qwen3[_stream]），
# *               屏蔽 generate / generate_qwen3 的接口分裂。
# *               重依赖（torch、tokenizer、模型）全部懒加载。
# * Requirement   : torch
# ***************************************************

__all__ = [
    "generate_text",
    "generate_text_stream",
]

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def _encode(prompt, family: str, tokenizer_model, tokenizer_file_path, device):
    """prompt 文本 -> token ids tensor"""
    import torch
    if family == "qwen3":
        from layers.tokenizers.qwen3.qwen3_tokenizer import Qwen3Tokenizer
        tokenizer = Qwen3Tokenizer(
            tokenizer_file_path=tokenizer_file_path,
            apply_chat_template=False,
            add_generation_prompt=False,
        )
        return torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0), tokenizer
    from layers.tokenizers.tokenization import text_to_token_ids, choose_tokenizer
    return (
        text_to_token_ids(prompt, tokenizer_model=tokenizer_model).to(device),
        choose_tokenizer(tokenizer_model=tokenizer_model),
    )


def _decode(token_ids, family: str, tokenizer_model, tokenizer):
    """token ids tensor -> 文本"""
    if family == "qwen3":
        return tokenizer.decode(token_ids.squeeze(0).tolist())
    from layers.tokenizers.tokenization import token_ids_to_text
    return token_ids_to_text(token_ids, tokenizer_model=tokenizer_model)


def generate_text(model, prompt: str, family: str = "gpt2", tokenizer_model: str | None = None,
                  tokenizer_file_path: str | None = None, max_new_tokens: int = 50,
                  context_length: int | None = None, temperature: float = 0.0,
                  top_k: int | None = None, top_p: float | None = None,
                  eos_token_id: int | None = None, use_cache: bool = False,
                  device: str = "cpu") -> str:
    """
    统一文本生成入口（批量/流式见 generate_text_stream 与 layers.inference）。

    Args:
        model: 已加载权重的模型（见 inference.loading.load_model）
        prompt: 输入文本
        family: 模型族（gpt2 / llama / qwen3），决定生成实现
        tokenizer_model: layers.tokenizers 的 tokenizer 名（gpt2/llama 必填）
        tokenizer_file_path: Qwen3 tokenizer.json 路径（family="qwen3" 必填，
            即 load_model 返回 extras["tokenizer_file_path"]）
        use_cache: KV cache（仅 qwen3 族支持；gpt2/llama 的 GPT 注意力未实现 cache）
    """
    tokenizer_model = tokenizer_model or ("tiktoken_gpt2_bpe" if family == "gpt2" else family)
    token_ids, tokenizer = _encode(prompt, family, tokenizer_model, tokenizer_file_path, device)
    if context_length is None:
        # qwen3 Model 持有 self.cfg；GPT-2 Model 用 pos_embed 容量
        context_length = (model.cfg.context_length if hasattr(model, "cfg")
                          else model.pos_embed.num_embeddings)
    ctx = context_length

    if family == "qwen3":
        from layers.inference import generate_qwen3
        out = generate_qwen3(
            model, token_ids, max_new_tokens=max_new_tokens, context_length=ctx,
            temperature=temperature, top_k=top_k, top_p=top_p,
            eos_token_id=eos_token_id, use_cache=use_cache,
        )
        # generate_qwen3 只返回新生成部分
        return tokenizer.decode(out.squeeze(0).tolist())

    if use_cache:
        raise ValueError(f"{family} 族的 GPT 注意力未实现 KV cache，请用 use_cache=False")
    from layers.inference import generate
    out = generate(
        model, token_ids, max_new_tokens=max_new_tokens, context_length=ctx,
        temperature=temperature, top_k=top_k, top_p=top_p,
        eos_token_id=eos_token_id, use_cache=False,
    )
    return _decode(out, family, tokenizer_model, tokenizer)


def generate_text_stream(model, prompt: str, family: str = "qwen3", **kwargs):
    """
    流式生成：逐 token yield 解码文本片段（目前仅 qwen3 族支持）。
    kwargs 同 generate_text。
    """
    if family != "qwen3":
        raise NotImplementedError("流式生成目前仅支持 qwen3 族（generate_qwen3_stream）")
    tokenizer_file_path = kwargs.pop("tokenizer_file_path", None)
    device = kwargs.pop("device", "cpu")
    token_ids, tokenizer = _encode(prompt, family, None, tokenizer_file_path, device)
    ctx = kwargs.pop("context_length", None) or model.cfg.context_length

    from layers.inference import generate_qwen3_stream
    for token in generate_qwen3_stream(model, token_ids, context_length=ctx, **kwargs):
        yield tokenizer.decode(token.squeeze(0).tolist())
