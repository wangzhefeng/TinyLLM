# -*- coding: utf-8 -*-

# ***************************************************
# * File        : speculative.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102423
# * Description : 投机解码（Speculative Decoding, 推断时扩展策略之一）：
# *               用小 draft 模型一次猜 num_draft_tokens 个 token，
# *               再由大 target 模型一次 forward 并行验证，接受最长匹配前缀，
# *               greedy 下输出与普通 greedy 解码逐 token 一致。
# * Link        : https://arxiv.org/abs/2211.17192
# * Requirement : torch
# ***************************************************

__all__ = [
    "speculative_generate",
]

# python libraries
from pathlib import Path

import torch

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def _reset_cache(model):
    if hasattr(model, "reset_kv_cache"):
        model.reset_kv_cache()


def _step(model, token: int, device) -> torch.Tensor:
    """以 KV cache 方式喂入单个 token，返回 (vocab_size,) 的下一位置 logits。"""
    x = torch.tensor([[token]], dtype=torch.long, device=device)
    return model(x, use_cache=True)[0, -1]


def _adjust_logits(logits: torch.Tensor, temperature: float, top_k: int | None) -> torch.Tensor:
    """与 sample_next_token 一致的 temperature/top_k 变换（采样模式下用）。"""
    if temperature is not None and temperature > 0.0:
        logits = logits / temperature
    if top_k is not None:
        top_k = min(top_k, logits.size(-1))
        top_logits, _ = torch.topk(logits, top_k, dim=-1)
        min_val = top_logits[..., -1:]
        logits = torch.where(logits < min_val, torch.full_like(logits, float("-inf")), logits)
    return logits


@torch.inference_mode()
def speculative_generate(model, draft_model, token_ids: torch.Tensor, max_new_tokens: int,
                         num_draft_tokens: int = 4, eos_token_id: int | None = None,
                         context_length: int | None = None, temperature: float = 0.0,
                         top_k: int | None = None) -> torch.Tensor:
    """投机解码生成（教学实现，batch=1，GPT-2 风格 forward 协议）。

    算法（每轮）：
        1. draft 模型以当前序列为前缀，连续生成最多 ``num_draft_tokens`` 个草稿 token；
        2. target 模型对草稿做一次并行 forward，得到每个草稿位置的验证分布；
        3. 验证：
           - greedy（temperature==0 且 top_k 为 None）：接受与 target argmax 匹配的
             最长草稿前缀，首个不匹配处丢弃其后草稿并改用 target argmax，
             该轮接受 (匹配数+1) 个 token；全部匹配时额外接受 target 的 bonus token；
           - 采样模式：第 i 个草稿以 min(1, p_target(t_i)/p_draft(t_i)) 概率接受，
             首个被拒位置从 normalize(max(p_target - p_draft, 0)) 重采样一个 token，
             其后草稿全部丢弃；全部接受时从 target 分布采样 bonus token；
        4. 遇到 eos_token_id：追加后终止；达到 max_new_tokens 终止。

    KV cache 一致性取舍（教学简化）：
        被拒绝的草稿 token 已写入 draft/target 的 cache，回滚 cache 容易出错，
        因此只要本轮发生拒绝，就 ``reset_kv_cache()`` 并用当前完整序列重新
        prefill 两个模型的 cache（O(T) 重填充）。全部接受时只需把 bonus token
        继续喂入两个 cache，无需重填。draft_model is model（自投机）时二者共享
        cache，验证前统一 reset + 整段重放，保证位置一致。

    Args:
        model: target（大）模型，forward(x, use_cache=True)，可选 reset_kv_cache()
        draft_model: draft（小）模型，协议同上；可与 model 为同一对象（自投机）
        token_ids: (1, T) prompt
        max_new_tokens: 最多新生成的 token 数
        num_draft_tokens: 每轮草稿 token 数上限
        eos_token_id: 生成到该 token（含）即停止
        context_length: 上下文窗口；None 时取 model.pos_embed.num_embeddings
        temperature: 0 表示 greedy（需 top_k 也为 None）
        top_k: 采样模式的 top-k 约束

    Returns:
        (1, T+L) 完整 token 序列（含 prompt），与 generate_simple 的返回约定一致
    """
    assert token_ids.size(0) == 1, "投机解码教学实现要求 batch=1"
    model.eval()
    draft_model.eval()
    if context_length is None:
        context_length = model.pos_embed.num_embeddings
    seq = token_ids[:, -context_length:].clone()
    device = seq.device
    same_model = draft_model is model
    greedy = (temperature is None or temperature <= 0.0) and top_k is None

    # 初始化：两个 cache 均以 prompt 预填
    _reset_cache(model)
    if not same_model:
        _reset_cache(draft_model)
    target_logits = model(seq, use_cache=True)[0, -1]          # target 对下一 token 的预测
    draft_logits = target_logits if same_model else draft_model(seq, use_cache=True)[0, -1]

    produced = 0
    while produced < max_new_tokens:
        remaining = max_new_tokens - produced
        k = min(num_draft_tokens, remaining)

        # ---------- 1. draft 阶段：连续生成 k 个草稿 token ----------
        drafts, draft_dists = [], []
        logits = draft_logits
        for _ in range(k):
            if greedy:
                tok = int(torch.argmax(logits))
                dist = None
            else:
                probs = torch.softmax(_adjust_logits(logits, temperature, top_k), dim=-1)
                tok = int(torch.multinomial(probs, num_samples=1))
                dist = probs
            drafts.append(tok)
            draft_dists.append(dist)
            if (eos_token_id is not None and tok == eos_token_id) or len(drafts) >= k:
                break
            logits = _step(draft_model, tok, device)

        # ---------- 2. target 验证：一次 forward 覆盖全部草稿位置 ----------
        if same_model:
            # 自投机共享 cache：draft 阶段已推进共享 cache，须重放整段保证验证位置正确
            _reset_cache(model)
            out = model(torch.cat([seq, torch.tensor([drafts], device=device)], dim=1),
                        use_cache=True)
            n_prompt = seq.size(1)
            verify_logits = out[0, n_prompt - 1: n_prompt - 1 + len(drafts)]  # 逐草稿位置的验证 logits
            bonus_logits = out[0, n_prompt - 1 + len(drafts)]                 # 末尾草稿之后的 bonus logits
        else:
            # target cache 仍停留在 seq，喂入草稿即可；验证 drafts[0] 用轮首缓存的 target_logits
            out = model(torch.tensor([drafts], dtype=torch.long, device=device), use_cache=True)
            verify_logits = torch.cat([target_logits.unsqueeze(0), out[0, :-1]], dim=0)
            bonus_logits = out[0, -1]

        # ---------- 3. 接受 / 拒绝 ----------
        n_matched = 0
        rejected = False
        if greedy:
            for i, tok in enumerate(drafts):
                if int(torch.argmax(verify_logits[i])) == tok:
                    n_matched += 1
                else:
                    rejected = True
                    break
            if rejected:
                correction = int(torch.argmax(verify_logits[n_matched]))
                accepted = drafts[:n_matched] + [correction]
            else:
                accepted = drafts + [int(torch.argmax(bonus_logits))]
        else:
            accepted = []
            for i, tok in enumerate(drafts):
                p = torch.softmax(_adjust_logits(verify_logits[i], temperature, top_k), dim=-1)
                q = draft_dists[i]
                accept_prob = min(1.0, float(p[tok]) / max(float(q[tok]), 1e-12))
                if float(torch.rand(1)) < accept_prob:
                    accepted.append(tok)
                    n_matched += 1
                else:
                    # 从 normalize(max(p_target - p_draft, 0)) 重采样
                    residual = (p - q).clamp_min(0)
                    dist = p if float(residual.sum()) <= 0 else residual / residual.sum()
                    accepted.append(int(torch.multinomial(dist, num_samples=1)))
                    rejected = True
                    break
            if not rejected:
                bonus = torch.softmax(_adjust_logits(bonus_logits, temperature, top_k), dim=-1)
                accepted.append(int(torch.multinomial(bonus, num_samples=1)))

        # ---------- 4. 追加与终止判定 ----------
        done = False
        if eos_token_id is not None and eos_token_id in accepted:
            accepted = accepted[:accepted.index(eos_token_id) + 1]
            done = True
        if len(accepted) > remaining:
            accepted = accepted[:remaining]
            done = True
        seq = torch.cat([seq, torch.tensor([accepted], dtype=torch.long, device=device)], dim=1)
        produced += len(accepted)
        if done or produced >= max_new_tokens:
            break

        # ---------- 5. 同步 KV cache ----------
        if rejected:
            # 拒绝后 cache 含有被丢弃的草稿：直接 reset + 整段重填（教学取舍，见 docstring）
            _reset_cache(model)
            if not same_model:
                _reset_cache(draft_model)
            target_logits = model(seq, use_cache=True)[0, -1]
            draft_logits = target_logits if same_model else draft_model(seq, use_cache=True)[0, -1]
        else:
            # 全部接受：cache 已含 seq+drafts，只需补喂 bonus token
            target_logits = _step(model, accepted[-1], device)
            draft_logits = target_logits if same_model else _step(draft_model, accepted[-1], device)

    return seq




# 测试代码 main 函数
def main():
    from types import SimpleNamespace

    from inference.generate_core import generate_simple
    from models.gpt2.gpt2_124M import Model

    torch.manual_seed(123)
    cfg = SimpleNamespace(
        vocab_size=64, context_length=32, embed_dim=32, d_ff=64,
        n_heads=4, n_layers=2, dropout=0.0, qkv_bias=True,
        dtype=torch.float32, is_train=False,
    )
    model = Model(cfg).eval()
    prompt = torch.tensor([[1, 2, 3, 4, 5]])

    greedy = generate_simple(model, prompt.clone(), max_new_tokens=8, context_length=32)
    # 自投机：draft 与 target 为同一模型
    spec = speculative_generate(model, model, prompt.clone(), max_new_tokens=8,
                                num_draft_tokens=4, context_length=32)
    logger.info(f"greedy     : {greedy.tolist()}")
    logger.info(f"speculative: {spec.tolist()}")
    logger.info(f"自投机 greedy 一致: {torch.equal(greedy, spec)}")

    # 独立 draft 模型（随机初始化）
    torch.manual_seed(456)
    draft = Model(cfg).eval()
    spec2 = speculative_generate(model, draft, prompt.clone(), max_new_tokens=8,
                                 num_draft_tokens=4, context_length=32)
    logger.info(f"独立 draft : {spec2.tolist()}")


if __name__ == "__main__":
    main()
