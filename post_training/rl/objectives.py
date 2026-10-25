import torch

from .core import completion_mask, token_logprobs


def _masked_normalize(values, mask, loss_norm):
    """按 ``loss_norm`` 对 (G, L) 张量归一化（mask 为 None 时全位置参与）。"""
    if loss_norm == "token_mean":
        if mask is not None:
            return (values * mask).sum() / mask.sum().clamp(min=1.0)
        return values.mean()
    if loss_norm == "seq_sum_mean":
        # Dr.GRPO 式长度偏差修正：序列内求和后对序列数取平均
        if mask is not None:
            return (values * mask).sum(dim=1).mean()
        return values.sum(dim=1).mean()
    raise ValueError(f"未知 loss_norm: {loss_norm!r}（可选 'token_mean' / 'seq_sum_mean'）")


def compute_clipped_policy_loss(
    model,
    samples,
    old_logprobs,
    advantages,
    prompt_len,
    clip_eps=0.2,
    kl_beta=0.0,
    ref_model=None,
    completion_lengths=None,
    clip_eps_high=None,
    kl_estimator="k3",
    loss_norm="token_mean",
    return_diagnostics=False,
):
    """GRPO/PPO 共用 clipped policy objective（token 级 ratio + PPO clip + 可选 KL）。

    ``L = -norm(min(ratio·A, clip(ratio)·A)) + kl_beta · norm(KL)``

    Args:
        model: 当前策略（可训练）
        samples: (G, T+L) 生成序列
        old_logprobs: (G, L) 采样时的旧策略 log-prob
        advantages: (G,) 优势（broadcast 到每个 token）
        prompt_len: prompt 长度（只对生成部分计算 loss）
        clip_eps: 比率下裁剪半径（下界 1 - clip_eps）
        kl_beta: KL 惩罚系数（0 = 不加）；需配合 ref_model
        ref_model: 冻结的参考策略
        completion_lengths: (G,) 真实 completion 长度；None 时全位置参与
        clip_eps_high: clip-higher 上裁剪半径（上界 1 + clip_eps_high）；
            None 时对称上界 1 + clip_eps（DAPO clip-higher）
        kl_estimator: "k3"（``exp(Δ) - Δ - 1``，无偏低方差）或
            "k1"（``0.5·Δ²``，计算最简）
        loss_norm: "token_mean"（有效 token 平均）或 "seq_sum_mean"
            （序列求和后对序列数取平均，Dr.GRPO 长度偏差修正）
        return_diagnostics: True 时返回 ``(loss, diagnostics)``，
            diagnostics 含 ``{"clip_frac", "kl", "ratio_mean"}``

    Returns:
        标量 loss；return_diagnostics=True 时为 (loss, diagnostics)
    """
    new_logprobs = token_logprobs(model, samples, start_pos=prompt_len)
    ratio = torch.exp(new_logprobs - old_logprobs)
    adv = advantages.unsqueeze(1)
    clip_low = 1 - clip_eps
    clip_high = 1 + (clip_eps if clip_eps_high is None else clip_eps_high)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, clip_low, clip_high) * adv
    token_obj = torch.min(unclipped, clipped)

    if completion_lengths is not None:
        mask = completion_mask(completion_lengths, new_logprobs.shape[1]).float()
    else:
        mask = None
    policy_loss = -_masked_normalize(token_obj, mask, loss_norm)

    kl_term = None
    if kl_beta > 0.0 and ref_model is not None:
        with torch.no_grad():
            ref_logprobs = token_logprobs(ref_model, samples, start_pos=prompt_len)
        log_ratio = new_logprobs - ref_logprobs
        if kl_estimator == "k3":
            kl = torch.exp(log_ratio) - log_ratio - 1.0
        elif kl_estimator == "k1":
            kl = 0.5 * log_ratio.square()
        else:
            raise ValueError(f"未知 kl_estimator: {kl_estimator!r}（可选 'k1' / 'k3'）")
        kl_term = _masked_normalize(kl, mask, loss_norm)
        policy_loss = policy_loss + kl_beta * kl_term

    if not return_diagnostics:
        return policy_loss

    # clip 分支生效 = min 取了 clipped：ratio 越上界且 A>0，或破下界且 A<0
    clip_active = ((ratio > clip_high) & (adv > 0)) | ((ratio < clip_low) & (adv < 0))
    if mask is not None:
        denom = mask.sum().clamp(min=1.0)
        clip_frac = float((clip_active.float() * mask).sum() / denom)
        ratio_mean = float((ratio * mask).sum() / denom)
    else:
        clip_frac = float(clip_active.float().mean())
        ratio_mean = float(ratio.mean())
    diagnostics = {
        "clip_frac": clip_frac,
        "kl": float(kl_term) if kl_term is not None else 0.0,
        "ratio_mean": ratio_mean,
    }
    return policy_loss, diagnostics
