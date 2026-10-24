"""后训练域默认路径/命名/import 无副作用回归测试（重构第一阶段）。"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]

SFT_IF_CONFIG = "post_training.sft.data_provider.instruction_follow.data_config"
SFT_CLF_CONFIG = "post_training.sft.data_provider.text_clf.data_config"
DPO_CONFIG = "post_training.preference.data_provider.dpo.data_config"


class TestDataConfigImportSideEffects(unittest.TestCase):
    """import data_config 不得创建 dataset/ 目录；data_dir 惰性创建。"""

    def _run_in_tmp_cwd(self, code: str) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        with tempfile.TemporaryDirectory() as tmp:
            return subprocess.run(
                [sys.executable, "-c", code],
                cwd=tmp, env=env, capture_output=True, text=True, timeout=300,
            )

    def test_import_does_not_create_dataset_dir(self):
        code = (
            f"import {SFT_IF_CONFIG}, {SFT_CLF_CONFIG}, {DPO_CONFIG}\n"
            "import pathlib, sys\n"
            "sys.exit(1 if pathlib.Path('dataset').exists() else 0)\n"
        )
        result = self._run_in_tmp_cwd(code)
        self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_data_dir_created_lazily_on_access(self):
        for module in (SFT_IF_CONFIG, SFT_CLF_CONFIG):
            with self.subTest(module=module):
                code = (
                    f"import {module} as m\n"
                    "import pathlib, sys\n"
                    "assert not pathlib.Path('dataset').exists(), 'import 不应建目录'\n"
                    "m.data_dir\n"
                    "sys.exit(0 if pathlib.Path('dataset').exists() else 1)\n"
                )
                result = self._run_in_tmp_cwd(code)
                self.assertEqual(result.returncode, 0, msg=result.stderr)


class TestDefaultDataPaths(unittest.TestCase):
    """SFT/DPO 默认数据路径统一指向 dataset/finetuning/。"""

    def test_instruction_follow_config_paths(self):
        from post_training.sft.data_provider.instruction_follow import data_config

        self.assertIn("dataset/finetuning", str(data_config.data_path))
        self.assertNotIn("dataset/finetune/", str(data_config.data_path))

    def test_text_clf_config_paths(self):
        from post_training.sft.data_provider.text_clf import data_config

        self.assertIn("dataset/finetuning", str(data_config.zip_data_path))
        self.assertIn("dataset/finetuning", str(data_config.tsv_file_path))

    def test_dpo_config_paths(self):
        from post_training.preference.data_provider.dpo import data_config

        for path in (
            data_config.instruction_data_path,
            data_config.instruction_data_with_preference_path,
        ):
            self.assertIn("dataset/finetuning/", path)

    def test_dpo_runner_argparse_defaults(self):
        with mock.patch.object(sys, "argv", ["run_gpt_dpo.py"]):
            from post_training.preference.model_rlhf.dpo import run_gpt_dpo

            args = run_gpt_dpo.args_parse()
        # DPO 默认数据路径在 dataset/finetuning/
        self.assertIn("dataset/finetuning/", args.data_path)
        # DPO 默认 SFT 权重为 gpt2-medium355M-sft.pth（355M 而非 335M）
        self.assertTrue(args.finetuned_model_path.endswith("gpt2-medium355M-sft.pth"))


class TestDpoOutputNaming(unittest.TestCase):
    """DPO 训练输出不得复用 -sft.pth 后缀。"""

    def test_dpo_output_named_dpo_pth(self):
        from post_training.preference.exp.exp_finetune_gpt_dpo import (
            ModelFinetuningPreference,
        )

        with tempfile.TemporaryDirectory() as tmp:
            exp = object.__new__(ModelFinetuningPreference)
            exp.args = SimpleNamespace(
                model_path=tmp, pretrained_model="gpt2-medium (355M)"
            )
            exp._get_model_path("setting", 0)
            self.assertEqual(exp.model_path.name, "gpt2-medium355M-dpo.pth")


class TestScriptLinks(unittest.TestCase):
    """低成本断链修复的回归保护。"""

    def test_train_grpo_compiles(self):
        import py_compile

        py_compile.compile(
            str(ROOT / "post_training/preference/model_rlhf/grpo"
                "/qwen_scheduler_grpo/train_grpo.py"),
            doraise=True,
        )

    def test_spectrum_script_config_exists(self):
        script = (ROOT / "post_training/peft/scripts"
                  "/spectrum_llama31_8b_qlora.sh").read_text()
        self.assertIn("post_training/peft/cfgs/llama-3-1-8b-qlora.yaml", script)
        self.assertNotIn("receipes/", script)
        self.assertTrue(
            (ROOT / "post_training/peft/cfgs/llama-3-1-8b-qlora.yaml").is_file()
        )

    def test_find_near_duplicates_script_target_exists(self):
        script = (ROOT / "post_training/common/scripts"
                  "/find_near_duplicates.sh").read_text()
        target = ("post_training/sft/data_provider/instruction_follow"
                  "/find_near_duplicates.py")
        self.assertIn(target, script)
        self.assertTrue((ROOT / target).is_file())


if __name__ == "__main__":
    unittest.main()
