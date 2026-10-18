# -*- coding: utf-8 -*-
"""
Gutenberg 预训练示例的集成测试（原 pre_training/data_provider/pgcorpus_gutenberg/tests.py）。

依赖上游 PyPI 包 `llms_from_scratch`，未安装时跳过；
在临时目录中运行，不污染仓库。
"""

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "pre_training" / "examples" / "pgcorpus_pretraining_simple.py"


class TestGutenbergExample(unittest.TestCase):

    @unittest.skipUnless(
        importlib.util.find_spec("llms_from_scratch"),
        "requires the upstream `llms_from_scratch` package",
    )
    def test_pretraining_simple_debug(self):
        sequence = "a b c d"
        repetitions = 1000
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "gutenberg" / "data"
            data_dir.mkdir(parents=True)
            (data_dir / "repeated_sequence.txt").write_text(sequence * repetitions)
            result = subprocess.run(
                [
                    sys.executable, str(SCRIPT),
                    "--debug", "true",
                    "--data_dir", str(data_dir),
                    "--output_dir", str(Path(tmp) / "model_checkpoints"),
                ],
                capture_output=True, text=True, cwd=tmp,
            )
            print(result.stdout)
            self.assertIn("Maximum GPU memory allocated", result.stdout)


if __name__ == "__main__":
    unittest.main()
