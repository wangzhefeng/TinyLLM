import io
import os
import re
import shutil
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F

from tools.distributed._common import dist as dist_utils
from tools.distributed._common.dataset import make_toy_dataset, make_toy_dataloaders
from tools.distributed._common.device import get_device, get_dist_device
from tools.distributed._common.metrics import log_step_metrics, peak_memory_mb
from tools.distributed._common.model import ModelArgs, ToyMLP, Transformer
from utils import ddp_utils

REPO_ROOT = Path(__file__).resolve().parents[2]
DIST_DIR = REPO_ROOT / "tools" / "distributed"
COMPOSE_DIR = DIST_DIR / "07-compose"

SLOW_TESTS = os.environ.get("TINYLLM_SLOW_TESTS")
SUBPROCESS_TIMEOUT = 180
MASTER_PORT_BASE = 29610


def run_subprocess(args, timeout=SUBPROCESS_TIMEOUT):
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def torchrun(script, *script_args, port, timeout=SUBPROCESS_TIMEOUT):
    return run_subprocess(
        [
            "-m", "torch.distributed.run",
            "--nproc_per_node=2",
            f"--master_port={port}",
            str(script),
            *script_args,
        ],
        timeout=timeout,
    )


def fail_message(name, proc):
    return (
        f"{name} exited with {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout[-2000:]}\n"
        f"--- stderr ---\n{proc.stderr[-2000:]}"
    )


class CommonDatasetTest(unittest.TestCase):
    def test_deterministic_per_seed(self):
        ds_a = make_toy_dataset(num_samples=64, seed=7)
        ds_b = make_toy_dataset(num_samples=64, seed=7)
        ds_c = make_toy_dataset(num_samples=64, seed=8)
        self.assertTrue(torch.equal(ds_a.features, ds_b.features))
        self.assertTrue(torch.equal(ds_a.labels, ds_b.labels))
        self.assertFalse(torch.equal(ds_a.features, ds_c.features))

    def test_shapes_and_label_range(self):
        ds = make_toy_dataset(num_samples=32, num_features=20, num_classes=2)
        self.assertEqual(len(ds), 32)
        features, label = ds[0]
        self.assertEqual(features.shape, (20,))
        self.assertIn(label.item(), (0, 1))

    def test_shared_rule_gives_high_test_accuracy(self):
        # rule_seed 保证 train/test 同一分类规则，线性模型应能学到 >0.9 的测试精度
        train_loader, test_loader = make_toy_dataloaders(batch_size=64, num_samples=512)
        model = torch.nn.Linear(20, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        for _ in range(5):
            for features, labels in train_loader:
                optimizer.zero_grad()
                loss = F.cross_entropy(model(features), labels)
                loss.backward()
                optimizer.step()
        correct = total = 0
        with torch.no_grad():
            for features, labels in test_loader:
                correct += (model(features).argmax(dim=1) == labels).sum().item()
                total += labels.numel()
        self.assertGreater(correct / total, 0.9)

    def test_dataloaders_split_and_batching(self):
        train_loader, test_loader = make_toy_dataloaders(batch_size=32, num_samples=256)
        self.assertEqual(len(train_loader), 256 // 32)
        self.assertEqual(len(test_loader), (256 // 4) // 32)
        self.assertFalse(getattr(test_loader.sampler, "shuffle", False))


class CommonModelTest(unittest.TestCase):
    def test_toy_mlp_output_shape(self):
        model = ToyMLP(num_features=20, num_classes=2)
        logits = model(torch.randn(4, 20))
        self.assertEqual(logits.shape, (4, 2))

    def test_transformer_output_shape(self):
        args = ModelArgs(n_layers=2, vocab_size=8, max_seq_len=16, dim=16, n_heads=4, dropout_p=0.0)
        model = Transformer(args)
        tokens = torch.randint(0, 8, (2, 16))
        self.assertEqual(model(tokens).shape, (2, 16, 8))

    def test_transformer_rejects_overlong_sequence(self):
        args = ModelArgs(max_seq_len=16, dropout_p=0.0)
        model = Transformer(args)
        with self.assertRaises(AssertionError):
            model(torch.randint(0, 8, (1, 17)))


class CommonDistTest(unittest.TestCase):
    def test_repo_root(self):
        self.assertEqual(dist_utils.REPO_ROOT, REPO_ROOT)
        self.assertTrue((dist_utils.REPO_ROOT / "pyproject.toml").exists())

    def test_rank_queries_without_process_group(self):
        self.assertFalse(dist_utils.is_dist())
        self.assertEqual(dist_utils.get_rank(), 0)
        self.assertEqual(dist_utils.get_world_size(), 1)
        self.assertEqual(dist_utils.get_local_rank(), 0)
        self.assertTrue(dist_utils.is_main_process())

    def test_rank0_print_only_on_main(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            dist_utils.rank0_print("hello")
        self.assertEqual(buf.getvalue().strip(), "hello")

    def test_get_checkpoint_dir_creates_directory(self):
        unit = "unittest-tmp"
        path = dist_utils.get_checkpoint_dir(unit)
        try:
            self.assertTrue(path.is_dir())
            self.assertEqual(path, REPO_ROOT / "results" / "checkpoints" / "distributed" / unit)
        finally:
            shutil.rmtree(path, ignore_errors=True)


class DeviceTest(unittest.TestCase):
    @unittest.skipIf(torch.cuda.is_available(), "CUDA 可用时不走 CPU 分支")
    def test_cpu_fallback(self):
        self.assertEqual(get_device(), torch.device("cpu"))
        expected = "mps" if torch.backends.mps.is_available() else "cpu"
        self.assertEqual(get_device(allow_mps=True).type, expected)
        # 分布式入口禁用 MPS（gloo 不支持），无 CUDA 时恒为 cpu
        self.assertEqual(get_dist_device(0), torch.device("cpu"))


class MetricsTest(unittest.TestCase):
    @unittest.skipIf(torch.cuda.is_available(), "CUDA 可用时 peak_memory_mb 不为 None")
    def test_peak_memory_none_on_cpu(self):
        self.assertIsNone(peak_memory_mb())

    def test_log_step_metrics_format(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            log_step_metrics(step=3, loss=0.5, tokens_per_sec=1000.0, extra="epoch 0")
        line = buf.getvalue().strip()
        self.assertIn("step     3", line)
        self.assertIn("loss 0.5000", line)
        self.assertIn("1000 tok/s", line)
        if not torch.cuda.is_available():
            self.assertIn("mem n/a", line)
        self.assertIn("epoch 0", line)


class DdpUtilsTest(unittest.TestCase):
    def setUp(self):
        self._env_patcher = mock.patch.dict(os.environ)
        self._env_patcher.start()
        for var in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
            os.environ.pop(var, None)

    def tearDown(self):
        self._env_patcher.stop()

    def test_non_distributed_defaults(self):
        self.assertFalse(ddp_utils.is_dist())
        self.assertEqual(ddp_utils.get_global_rank(), 0)
        self.assertEqual(ddp_utils.get_world_size(), 1)
        self.assertEqual(ddp_utils.get_local_rank(), 0)
        self.assertTrue(ddp_utils.is_master())

    def test_env_var_overrides(self):
        os.environ["RANK"] = "3"
        os.environ["WORLD_SIZE"] = "8"
        self.assertEqual(ddp_utils.get_global_rank(), 3)
        self.assertEqual(ddp_utils.get_world_size(), 8)
        # LOCAL_RANK 与 is_master 仅在进程组已初始化时看环境变量，
        # 非分布式下分别恒为 0 / True
        os.environ["LOCAL_RANK"] = "2"
        self.assertEqual(ddp_utils.get_local_rank(), 0)
        self.assertTrue(ddp_utils.is_master())


class StructureTest(unittest.TestCase):
    ENTRY_FILES = {
        "00-single-gpu": ["main.py"],
        "01-dp": ["main.py"],
        "02-ddp": ["multigpu_torchrun.py", "multigpu_spawn.py", "multinode.py"],
        "03-zero": ["train.py", "verify_equivalence.py", "run_all.sh"],
        "04-fsdp": ["train.py", "run.sh"],
        "05-tp": ["main.py", "verify_equivalence.py", "run.sh"],
        "06-pp": ["main.py", "run.sh"],
        "08-frameworks": [
            "accelerate_example/main.py",
            "accelerate_example/run.sh",
            "deepspeed_zero/main.py",
        ],
    }

    def test_entry_files_exist(self):
        for unit, files in self.ENTRY_FILES.items():
            for name in files:
                self.assertTrue((DIST_DIR / unit / name).is_file(), f"{unit}/{name} 缺失")
        if COMPOSE_DIR.is_dir():
            self.assertTrue(any(COMPOSE_DIR.glob("*.py")), "07-compose 存在但无入口脚本")

    def test_shell_scripts_have_no_dead_cuda_visible_devices_line(self):
        # 单独一行的 `CUDA_VISIBLE_DEVICES=...` 不导出也不作用于任何命令，是死代码
        dead_line = re.compile(r"^\s*CUDA_VISIBLE_DEVICES=\S*\s*$")
        for script in DIST_DIR.rglob("*.sh"):
            for lineno, line in enumerate(script.read_text().splitlines(), 1):
                self.assertFalse(
                    dead_line.match(line),
                    f"{script.relative_to(REPO_ROOT)}:{lineno} 存在孤立的 CUDA_VISIBLE_DEVICES 赋值",
                )


class SingleProcessSmokeTest(unittest.TestCase):
    """快速子进程冒烟：单进程入口与 CUDA-only 入口的优雅报错。"""

    def test_00_single_gpu_trains(self):
        proc = run_subprocess([str(DIST_DIR / "00-single-gpu" / "main.py"), "--epochs", "1"])
        self.assertEqual(proc.returncode, 0, fail_message("00-single-gpu", proc))
        self.assertIn("loss", proc.stdout)

    @unittest.skipIf(torch.cuda.is_available(), "CUDA 环境下 01-dp 可直接运行")
    def test_01_dp_fails_gracefully_without_cuda(self):
        proc = run_subprocess([str(DIST_DIR / "01-dp" / "main.py"), "--epochs", "1"])
        self.assertNotEqual(proc.returncode, 0, "01-dp 在无 CUDA 时应非零退出")
        self.assertIn("需要 CUDA", proc.stderr)

    @unittest.skipIf(torch.cuda.is_available(), "CUDA 环境下 04-fsdp 可直接运行")
    def test_04_fsdp_fails_gracefully_without_cuda(self):
        proc = run_subprocess([str(DIST_DIR / "04-fsdp" / "train.py"), "--steps", "1"])
        self.assertNotEqual(proc.returncode, 0, "04-fsdp 在无 CUDA 时应非零退出")
        self.assertIn("需要 CUDA", proc.stderr)

    @unittest.skipUnless(COMPOSE_DIR.is_dir(), "07-compose 尚不存在（并行开发中）")
    @unittest.skipIf(torch.cuda.is_available(), "CUDA 环境下 fsdp_tp 可直接运行")
    def test_07_compose_fsdp_tp_fails_gracefully_without_cuda(self):
        script = COMPOSE_DIR / "fsdp_tp.py"
        if not script.is_file():
            self.skipTest("07-compose/fsdp_tp.py 尚不存在")
        proc = run_subprocess([str(script)])
        self.assertNotEqual(proc.returncode, 0, "07-compose/fsdp_tp.py 在无 CUDA 时应非零退出")


@unittest.skipUnless(SLOW_TESTS, "多进程 gloo 冒烟：设 TINYLLM_SLOW_TESTS=1 启用")
class DistributedSmokeTest(unittest.TestCase):
    def test_02_ddp_torchrun(self):
        proc = torchrun(
            DIST_DIR / "02-ddp" / "multigpu_torchrun.py",
            "--epochs", "1", "--save-every", "0",
            port=MASTER_PORT_BASE + 1,
        )
        self.assertEqual(proc.returncode, 0, fail_message("02-ddp", proc))
        self.assertIn("loss", proc.stdout)

    def test_03_zero_stage1(self):
        proc = torchrun(
            DIST_DIR / "03-zero" / "train.py",
            "--stage", "1", "--epochs", "1", "--steps", "8",
            port=MASTER_PORT_BASE + 2,
        )
        self.assertEqual(proc.returncode, 0, fail_message("03-zero stage 1", proc))
        self.assertIn("stage 1", proc.stdout)

    def test_05_tp_verify_equivalence(self):
        proc = torchrun(
            DIST_DIR / "05-tp" / "verify_equivalence.py",
            port=MASTER_PORT_BASE + 3,
        )
        self.assertEqual(proc.returncode, 0, fail_message("05-tp verify", proc))
        self.assertIn("PASS", proc.stdout)

    def test_06_pp_verify(self):
        proc = torchrun(
            DIST_DIR / "06-pp" / "main.py",
            "--verify", "--steps", "5",
            port=MASTER_PORT_BASE + 4,
        )
        self.assertEqual(proc.returncode, 0, fail_message("06-pp verify", proc))
        self.assertIn("verify PASS", proc.stdout)

    def test_08_accelerate_torchrun(self):
        proc = torchrun(
            DIST_DIR / "08-frameworks" / "accelerate_example" / "main.py",
            "--epochs", "1",
            port=MASTER_PORT_BASE + 5,
        )
        self.assertEqual(proc.returncode, 0, fail_message("08-accelerate", proc))
        self.assertIn("test acc", proc.stdout)


if __name__ == "__main__":
    unittest.main()
