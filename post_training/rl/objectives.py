import torch

from .core import completion_mask, token_logprobs


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
):
    new_logprobs = token_logprobs(model, samples, start_pos=prompt_len)
    ratio = torch.exp(new_logprobs - old_logprobs)
    adv = advantages.unsqueeze(1)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    token_obj = torch.min(unclipped, clipped)

    if completion_lengths is not None:
        mask = completion_mask(completion_lengths, new_logprobs.shape[1]).float()
        policy_loss = -(token_obj * mask).sum() / mask.sum().clamp(min=1.0)
    else:
        mask = None
        policy_loss = -token_obj.mean()

    if kl_beta > 0.0 and ref_model is not None:
        with torch.no_grad():
            ref_logprobs = token_logprobs(ref_model, samples, start_pos=prompt_len)
        log_ratio = new_logprobs - ref_logprobs
        kl = torch.exp(log_ratio) - log_ratio - 1.0
        if mask is not None:
            kl_term = (kl * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            kl_term = kl.mean()
        policy_loss = policy_loss + kl_beta * kl_term
    return policy_loss
