import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

NANO_VLLM = Path(__file__).resolve().parents[2] / "inference" / "serving" / "nano_vllm"

# 在子进程中运行：nano_vllm 模块会改写 sys.path（PKG_ROOT 优先），
# 隔离以避免遮蔽仓库根目录的同名包（config/ utils/ 等）
LOAD_MODEL_SCRIPT = textwrap.dedent("""
    import sys
    from pathlib import Path
    PKG = sys.argv[1]
    sys.path.insert(0, PKG)
    import torch
    from safetensors.torch import save_file
    from config import Config
    from models.qwen3_for_causal_lm import Qwen3ForCausalLM
    from utils.loader import load_model

    tmp = sys.argv[2]
    cfg = Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                 num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                 head_dim=8, max_position_embeddings=64, max_model_len=64,
                 max_num_seqs=2, num_kvcache_blocks=8, eos=0)
    model = Qwen3ForCausalLM(cfg)

    # 构造 HF 命名风格的拆分权重(q/k/v 与 gate/up 分开存)
    sd = {
        "model.embed_tokens.weight": torch.arange(32 * 16, dtype=torch.float32).reshape(32, 16),
        "model.layers.0.self_attn.q_proj.weight": torch.full((16, 16), 1.0),
        "model.layers.0.self_attn.k_proj.weight": torch.full((8, 16), 2.0),
        "model.layers.0.self_attn.v_proj.weight": torch.full((8, 16), 3.0),
        "model.layers.0.self_attn.o_proj.weight": torch.full((16, 16), 4.0),
        "model.layers.0.mlp.gate_proj.weight": torch.full((32, 16), 5.0),
        "model.layers.0.mlp.up_proj.weight": torch.full((32, 16), 6.0),
        "model.layers.0.mlp.down_proj.weight": torch.full((16, 32), 7.0),
        "model.layers.0.self_attn.q_norm.weight": torch.ones(8),
        "model.layers.0.self_attn.k_norm.weight": torch.ones(8),
        "model.layers.0.input_layernorm.weight": torch.ones(16),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(16),
        "model.norm.weight": torch.ones(16),
        "lm_head.weight": torch.full((32, 16), 8.0),
    }
    save_file(sd, str(Path(tmp) / "model.safetensors"))
    load_model(model, tmp)

    # 直接拷贝
    assert torch.equal(model.model.embed_tokens.weight, sd["model.embed_tokens.weight"])
    # qkv 打包: q(16) + k(8) + v(8) = 32 行
    qkv = model.model.layers[0].self_attn.qkv_proj.weight
    assert torch.equal(qkv[:16], sd["model.layers.0.self_attn.q_proj.weight"])
    assert torch.equal(qkv[16:24], sd["model.layers.0.self_attn.k_proj.weight"])
    assert torch.equal(qkv[24:32], sd["model.layers.0.self_attn.v_proj.weight"])
    # gate_up 打包
    gate_up = model.model.layers[0].mlp.gate_up_proj.weight
    assert torch.equal(gate_up[:32], sd["model.layers.0.mlp.gate_proj.weight"])
    assert torch.equal(gate_up[32:], sd["model.layers.0.mlp.up_proj.weight"])
    print("LOAD_MODEL_OK")
""")


class TestNanoVllmLoadModel(unittest.TestCase):
    def test_load_model_from_safetensors_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, "-c", LOAD_MODEL_SCRIPT, str(NANO_VLLM), tmp],
                capture_output=True, text=True, timeout=120,
            )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("LOAD_MODEL_OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
