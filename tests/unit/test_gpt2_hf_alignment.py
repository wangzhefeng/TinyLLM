"""GPT-2 手写实现与 HuggingFace 官方实现的端到端对齐测试。

对照 NamiGPT 的双档对齐测试法：
- 随机权重：tiny 配置 + 固定种子，双方前向 logits 与逐步 argmax 对齐（CI 常跑，无下载）；
- 真实权重：本地存在官方 124M 缓存时加载真实权重，对齐 logits 与贪心续写。
区别于 test_gpt2_loading.py（合成 state dict 的键名映射），这里验证的是
权重搬运之后「前向计算结果」与官方实现一致。
"""
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from models.gpt2.gpt2_124M import Model
from models.io.gpt2.openai_gpt2_weights_load_hf import load_weights_hf

try:
    from transformers import GPT2Config, GPT2LMHeadModel
    transformers_available = True
except ImportError:
    transformers_available = False

VOCAB, CTX, EMB, D_FF, N_LAYERS = 128, 16, 32, 64, 2

GPT2_CACHE = Path("downloaded_models/gpt2_model/models--openai-community--gpt2")

# tiktoken gpt2 编码 "Every effort moves"（由本地 tiktoken 实测）；
# 对齐断言只要求两侧吃同一串 token id，编码本身不影响测试有效性
PROMPT_IDS = [6109, 3626, 6100]


def _tiny_cfg():
    return SimpleNamespace(
        vocab_size=VOCAB, context_length=CTX, embed_dim=EMB, d_ff=D_FF,
        n_heads=4, n_layers=N_LAYERS, dropout=0.0, qkv_bias=True,
        dtype=torch.float32, is_train=False,
    )


def _hf_tiny_model(seed=123):
    """tiny 维度的官方 GPT2LMHeadModel（gelu_new + LN eps 1e-5 与手写实现一致）。"""
    torch.manual_seed(seed)
    cfg = GPT2Config(
        vocab_size=VOCAB, n_positions=CTX, n_embd=EMB, n_layer=N_LAYERS,
        n_head=4, n_inner=D_FF,
        attn_pdrop=0.0, embd_pdrop=0.0, resid_pdrop=0.0,
    )
    return GPT2LMHeadModel(cfg).eval()


def _as_base_model(hf_model):
    """GPT2LMHeadModel.state_dict 带 ``transformer.`` 前缀，剥掉以适配
    load_weights_hf 的基础 GPT2Model 键名约定（wpe/h.*/ln_f）。"""
    d = {
        key[len("transformer."):]: value
        for key, value in hf_model.state_dict().items()
        if key.startswith("transformer.")
    }
    return SimpleNamespace(state_dict=lambda: d)


def _tinyllm_from_hf(hf_model, n_layers=N_LAYERS):
    gpt = Model(_tiny_cfg()).eval()
    load_weights_hf(gpt, _as_base_model(hf_model), {"n_layers": n_layers})
    return gpt.eval()


def _snapshot_dir():
    snaps = sorted(GPT2_CACHE.glob("snapshots/*"))
    return snaps[-1] if snaps else None


@unittest.skipUnless(transformers_available, "transformers 未安装")
class TestRandomWeightAlignment(unittest.TestCase):
    """随机权重 + 固定种子：前向 logits 与逐步贪心 argmax 完全一致。"""

    def test_logits_allclose(self):
        hf = _hf_tiny_model()
        gpt = _tinyllm_from_hf(hf)
        torch.manual_seed(7)
        ids = torch.randint(0, VOCAB, (2, 8))
        with torch.no_grad():
            logits_hf = hf(ids).logits
            logits_ours = gpt(ids)
        torch.testing.assert_close(logits_ours, logits_hf, atol=1e-5, rtol=1e-5)

    def test_stepwise_greedy_argmax_matches(self):
        hf = _hf_tiny_model()
        gpt = _tinyllm_from_hf(hf)
        torch.manual_seed(7)
        ids = torch.randint(0, VOCAB, (1, 6))
        with torch.no_grad():
            for _ in range(5):
                nxt_hf = torch.argmax(hf(ids).logits[:, -1], dim=-1, keepdim=True)
                nxt_ours = torch.argmax(gpt(ids)[:, -1], dim=-1, keepdim=True)
                self.assertTrue(torch.equal(nxt_hf, nxt_ours))
                ids = torch.cat([ids, nxt_ours], dim=1)


@unittest.skipUnless(
    transformers_available and _snapshot_dir() is not None,
    "需要本地 gpt2 官方权重缓存（downloaded_models/gpt2_model）",
)
class TestRealWeightAlignment(unittest.TestCase):
    """真实 124M 权重：logits 对齐 + 贪心续写逐 token 一致。"""

    def test_real_logits_and_greedy_match(self):
        hf = GPT2LMHeadModel.from_pretrained(
            _snapshot_dir(), local_files_only=True, torch_dtype=torch.float32,
        ).eval()
        cfg = SimpleNamespace(
            vocab_size=hf.config.vocab_size,
            context_length=hf.config.n_positions,
            embed_dim=hf.config.n_embd,
            d_ff=hf.config.n_embd * 4,
            n_heads=hf.config.n_head,
            n_layers=hf.config.n_layer,
            dropout=0.0, qkv_bias=True, dtype=torch.float32, is_train=False,
        )
        gpt = Model(cfg).eval()
        load_weights_hf(gpt, _as_base_model(hf), {"n_layers": cfg.n_layers})

        ids = torch.tensor([PROMPT_IDS])
        with torch.no_grad():
            logits_hf = hf(ids).logits
            logits_ours = gpt(ids)
        torch.testing.assert_close(logits_ours, logits_hf, atol=1e-4, rtol=1e-4)

        seq_hf, seq_ours = ids.clone(), ids.clone()
        with torch.no_grad():
            for _ in range(8):
                seq_hf = torch.cat(
                    [seq_hf, torch.argmax(hf(seq_hf).logits[:, -1], dim=-1, keepdim=True)], dim=1,
                )
                seq_ours = torch.cat(
                    [seq_ours, torch.argmax(gpt(seq_ours)[:, -1], dim=-1, keepdim=True)], dim=1,
                )
        self.assertTrue(torch.equal(seq_hf, seq_ours))


if __name__ == "__main__":
    unittest.main()
