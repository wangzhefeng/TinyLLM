# -*- coding: utf-8 -*-

# ***************************************************
# * File        : test_nano_vllm_tp.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-20
# * Version     : 1.0.102020
# * Description : nano_vllm 多进程张量并行(TP)一致性测试：
# *               同一微型随机 Qwen3 分别以 tp=1 与 tp=2 跑贪心生成，
# *               断言两者输出 token 完全一致(gloo 后端, CPU 可跑)。
# *               在子进程中运行，隔离 nano_vllm 对 sys.path 的改写。
# * Link        : https://github.com/GeeeekExplorer/nano-vllm
# * Requirement : torch >= 2.0
# ***************************************************

import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

NANO_VLLM = Path(__file__).resolve().parents[2] / "inference" / "serving" / "nano_vllm"

# 注意：spawn 的 worker 子进程会重新 import __main__，
# 因此脚本主体必须放在 if __name__ == "__main__" 保护内
TP_SCRIPT = textwrap.dedent("""
    import sys
    from pathlib import Path
    PKG = sys.argv[1]
    sys.path.insert(0, PKG)

    from config import Config
    from sampling_params import SamplingParams
    from engine.llm_engine import LLMEngine


    def build_engine(tp: int) -> LLMEngine:
        config = Config(
            vocab_size=64, hidden_size=128, intermediate_size=256,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=32, max_position_embeddings=256, max_model_len=96,
            max_num_seqs=4, max_num_batched_tokens=24,
            eos=0, kvcache_block_size=16, num_kvcache_blocks=64,
            tensor_parallel_size=tp,
        )
        return LLMEngine(config, verbose=False)


    def main():
        prompts = [
            [3, 5, 9, 1, 22, 17, 8, 4, 6, 11, 30, 2, 7, 15, 19, 21, 13, 25, 27, 31, 40, 41, 42, 43, 44, 45],
            [10, 20, 30, 40, 50, 60, 1, 2, 3, 4],
            [7, 14, 21, 28, 35, 42, 49, 56, 63, 1, 2, 3, 4, 5],
        ]
        sp = SamplingParams(temperature=0.0, max_tokens=8)  # 贪心，保证确定性
        e1 = build_engine(1)
        out1 = [o["token_ids"] for o in e1.generate(prompts, sp)]
        e2 = build_engine(2)
        out2 = [o["token_ids"] for o in e2.generate(prompts, sp)]
        e2.exit()
        print("TP=1 outputs:", out1)
        print("TP=2 outputs:", out2)
        assert out1 == out2, f"TP=1 与 TP=2 输出不一致: {out1} vs {out2}"
        print("PASS: TP=1 与 TP=2 贪心生成输出完全一致")


    if __name__ == "__main__":
        main()
""")


class TestNanoVllmTP(unittest.TestCase):

    def test_tp1_tp2_consistency(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "tp_check.py"
            script.write_text(TP_SCRIPT, encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(script), str(NANO_VLLM)],
                capture_output=True, text=True, timeout=600,
            )
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            self.assertEqual(result.returncode, 0, f"TP 一致性脚本失败:\n{result.stderr}")
            self.assertIn("PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()
