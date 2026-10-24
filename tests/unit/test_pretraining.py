# -*- coding: utf-8 -*-
"""
预训练域单元测试（不跑真实预训练）：

- LR 调度（附录 D warmup + cosine）
- 梯度累积等价性
- 词表 padding / embedding 扩容 / 特殊 token 扩展（ch05/08-10）
- 滑动窗口数据集（ch02）
- checkpoint 保存/恢复闭环
- legacy 入口可导入
"""

import importlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

from utils.learning_rate import warmup_cosine_lr, set_optimizer_lr
from layers.tokenizers.extension import (
    pad_vocab_size,
    resize_token_embeddings,
    ExtendedTokenizer,
)
from pre_training.data_provider.data_loader import data_split, LLMDataset, create_dataloader
from pre_training.experiments.exp_pretrain_gpt2_124M import Model_Pretrain
from tools.checkpoint import CheckpointManager


class _CharTokenizer:
    """测试用字符级 tokenizer：encode/decode + n_vocab。"""

    n_vocab = 256

    def encode(self, text):
        return [ord(c) % 256 for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class TestLRSchedule(unittest.TestCase):
    """附录 D：线性 warmup + 余弦衰减。"""

    def test_warmup_starts_at_initial_lr(self):
        lr = warmup_cosine_lr(global_step=0, total_steps=100, warmup_steps=20,
                              peak_lr=1e-3, initial_lr=1e-4, min_lr=1e-6)
        self.assertAlmostEqual(lr, 1e-4)

    def test_warmup_increases_linearly(self):
        lrs = [warmup_cosine_lr(s, 100, 20, 1e-3, 1e-4, 1e-6) for s in range(20)]
        self.assertTrue(all(b > a for a, b in zip(lrs, lrs[1:])))

    def test_peak_at_warmup_end(self):
        lr = warmup_cosine_lr(global_step=20, total_steps=100, warmup_steps=20,
                              peak_lr=1e-3, initial_lr=1e-4, min_lr=1e-6)
        self.assertAlmostEqual(lr, 1e-3)

    def test_decays_to_min_lr(self):
        lr = warmup_cosine_lr(global_step=100, total_steps=100, warmup_steps=20,
                              peak_lr=1e-3, initial_lr=1e-4, min_lr=1e-6)
        self.assertAlmostEqual(lr, 1e-6)

    def test_set_optimizer_lr(self):
        model = nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        set_optimizer_lr(optimizer, 5e-5)
        self.assertTrue(all(g["lr"] == 5e-5 for g in optimizer.param_groups))


class TestGradAccumulation(unittest.TestCase):
    """梯度累积：N 个 micro-batch 的 loss/N 反向传播 == 整批一次反向传播。"""

    def test_accumulated_grads_match_full_batch(self):
        torch.manual_seed(42)
        model_a = nn.Linear(8, 8)
        model_b = nn.Linear(8, 8)
        model_b.load_state_dict(model_a.state_dict())
        x = torch.randn(4, 8)
        y = torch.randn(4, 8)
        criterion = nn.MSELoss()

        # 整批
        loss_full = criterion(model_a(x), y)
        loss_full.backward()

        # 2 个 micro-batch 累积
        accum_steps = 2
        for i in range(accum_steps):
            xi, yi = x[i * 2:(i + 1) * 2], y[i * 2:(i + 1) * 2]
            loss_i = criterion(model_b(xi), yi) / accum_steps
            loss_i.backward()

        for pa, pb in zip(model_a.parameters(), model_b.parameters()):
            self.assertTrue(torch.allclose(pa.grad, pb.grad, atol=1e-6))


class TestVocabPaddingAndResize(unittest.TestCase):
    """ch05/09 + ch05/10：词表对齐与 embedding 扩容。"""

    def test_pad_vocab_size(self):
        self.assertEqual(pad_vocab_size(50257, 64), 50304)
        self.assertEqual(pad_vocab_size(50304, 64), 50304)
        self.assertEqual(pad_vocab_size(50257, 0), 50257)

    def _tiny_gpt2(self, vocab_size=50):
        from models.gpt2.gpt2_124M import Model
        cfg = SimpleNamespace(
            vocab_size=vocab_size, context_length=8, embed_dim=16, d_ff=32,
            n_heads=2, n_layers=1, dropout=0.0, qkv_bias=True,
            dtype=torch.float32, is_train=False,
        )
        return Model(cfg)

    def test_resize_token_embeddings(self):
        model = self._tiny_gpt2(vocab_size=50)
        old_weight = model.tok_embed.weight.detach().clone()
        resize_token_embeddings(model, 64)
        self.assertEqual(model.tok_embed.weight.shape, (64, 16))
        self.assertEqual(model.out_head.weight.shape, (64, 16))
        # 旧权重复制保留
        self.assertTrue(torch.equal(model.tok_embed.weight[:50], old_weight))
        # 前向兼容新 id
        x = torch.randint(0, 64, (1, 4))
        self.assertEqual(model(x).shape, (1, 4, 64))

    def test_resize_keeps_weight_tying(self):
        embed = nn.Embedding(10, 8)
        model = SimpleNamespace()
        # 用最小 nn.Module 模拟 weight tying
        class _Tied(nn.Module):
            def __init__(self):
                super().__init__()
                self.tok_embed = nn.Embedding(10, 8)
                self.out_head = nn.Linear(8, 10, bias=False)
                self.out_head.weight = self.tok_embed.weight
        tied_model = _Tied()
        resize_token_embeddings(tied_model, 16)
        self.assertIs(tied_model.out_head.weight, tied_model.tok_embed.weight)

    def test_resize_rejects_shrink(self):
        model = self._tiny_gpt2(vocab_size=50)
        with self.assertRaises(ValueError):
            resize_token_embeddings(model, 32)


class TestExtendedTokenizer(unittest.TestCase):
    """ch05/09：特殊 token 扩展。"""

    def test_encode_decode_roundtrip(self):
        base = _CharTokenizer()
        tok = ExtendedTokenizer(base, {"<|im_start|>": None, "<|im_end|>": None})
        self.assertEqual(tok.n_vocab, 258)
        ids = tok.encode("a<|im_start|>b<|im_end|>c")
        self.assertEqual(ids, [97, 256, 98, 257, 99])
        self.assertEqual(tok.decode(ids), "a<|im_start|>b<|im_end|>c")

    def test_explicit_ids_and_conflict_check(self):
        base = _CharTokenizer()
        tok = ExtendedTokenizer(base, {"<|eos|>": 300})
        self.assertEqual(tok.encode("<|eos|>"), [300])
        with self.assertRaises(ValueError):
            ExtendedTokenizer(base, {"<|bad|>": 10})


class TestDataset(unittest.TestCase):
    """ch02：滑动窗口数据集。"""

    def test_data_split(self):
        train, valid = data_split("x" * 100, train_ratio=0.8)
        self.assertEqual((len(train), len(valid)), (80, 20))

    def test_sliding_window_shapes_and_shift(self):
        text = "abcdefghijklmnopqrstuvwxyz" * 4  # 104 chars
        ds = LLMDataset(text=text, tokenizer=_CharTokenizer(), max_len=8, stride=8,
                        num_examples=1000, flag="train", dtype=torch.long, device=None)
        self.assertEqual(ds.input_blocks.shape[1], 8)
        self.assertEqual(ds.target_blocks.shape[1], 8)
        # target 是 input 右移一位
        x, y = ds[0]
        self.assertTrue(torch.equal(x[1:], y[:-1]))

    def test_blocks_stay_on_cpu(self):
        """数据集 tensor 保留在 CPU（与 num_workers/pin_memory 组合兼容）。"""
        text = "abcdefghijklmnopqrstuvwxyz" * 4
        ds = LLMDataset(text=text, tokenizer=_CharTokenizer(), max_len=8, stride=8,
                        num_examples=1000, flag="train", dtype=torch.long,
                        device=torch.device("cpu"))
        self.assertEqual(ds.input_blocks.device.type, "cpu")
        self.assertEqual(ds.target_blocks.device.type, "cpu")

    def test_num_examples_boundary(self):
        """num_examples 是上限而不是上限+1（修复前 > 判断会多取一条）。"""
        text = "abcdefghijklmnopqrstuvwxyz" * 4
        ds = LLMDataset(text=text, tokenizer=_CharTokenizer(), max_len=8, stride=8,
                        num_examples=3, flag="train", dtype=torch.long, device=None)
        self.assertEqual(len(ds), 3)

    def test_create_dataloader_local_cpu(self):
        """本地文件走通 create_dataloader；CPU 环境下 pin_memory 必须为 False。"""
        with tempfile.TemporaryDirectory() as tmp:
            data_file = "toy.txt"
            with open(Path(tmp) / data_file, "w", encoding="utf-8") as f:
                f.write("abcdefghijklmnopqrstuvwxyz" * 40)
            ds, loader = create_dataloader(
                data_source="local", url=None, data_path=tmp, data_file=data_file,
                flag="train", train_ratio=0.9, tokenizer=_CharTokenizer(),
                batch_size=2, max_len=8, stride=8, num_workers=0, device=None,
            )
            batch_x, batch_y = next(iter(loader))
            self.assertEqual(batch_x.shape, (2, 8))
            self.assertEqual(batch_x.device.type, "cpu")
            if not torch.cuda.is_available():
                self.assertFalse(loader.pin_memory)


class TestDDPUtils(unittest.TestCase):
    """ddp_utils：rank/world/local_rank 来源与 init_dist 签名。"""

    def test_local_rank_from_env_without_init(self):
        """torchrun 在进程组初始化前注入 LOCAL_RANK，未初始化时也必须读到。"""
        from utils.ddp_utils import get_local_rank
        self.assertFalse(torch.distributed.is_initialized())
        with mock.patch.dict(os.environ, {"LOCAL_RANK": "3"}):
            self.assertEqual(get_local_rank(), 3)

    def test_local_rank_default_zero(self):
        from utils.ddp_utils import get_local_rank
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LOCAL_RANK", None)
            self.assertEqual(get_local_rank(), 0)

    def test_rank_and_world_size_from_env(self):
        from utils.ddp_utils import get_global_rank, get_world_size
        with mock.patch.dict(os.environ, {"RANK": "2", "WORLD_SIZE": "4"}):
            self.assertEqual(get_global_rank(), 2)
            self.assertEqual(get_world_size(), 4)

    def test_init_dist_accepts_local_rank(self):
        """init_dist 第三参数 local_rank 可选，不破坏旧的两参数调用。"""
        import inspect
        from utils.ddp_utils import init_dist
        params = list(inspect.signature(init_dist).parameters)
        self.assertEqual(params, ["rank", "world_size", "local_rank"])


class TestWrapModel(unittest.TestCase):
    """DP/DDP 包装：DP 使用可见设备内的逻辑索引；CPU/单卡不包装。"""

    def _bare_exp(self, **args_overrides):
        exp = Model_Pretrain.__new__(Model_Pretrain)
        exp.args = SimpleNamespace(
            use_gpu=False, use_dp=0, use_ddp=0, device_ids=[0],
        )
        exp.local_rank = 0
        for k, v in args_overrides.items():
            setattr(exp.args, k, v)
        return exp

    def test_dp_uses_logical_device_ids(self):
        """CUDA_VISIBLE_DEVICES=2,3 时 DP device_ids 必须是逻辑编号 [0, 1]。"""
        exp = self._bare_exp(use_gpu=True, use_dp=1, device_ids=[2, 3])
        model = nn.Linear(2, 1)
        with mock.patch("pre_training.experiments.exp_basic.nn.DataParallel") as dp_mock:
            dp_mock.side_effect = lambda m, device_ids: ("dp", m, device_ids)
            wrapped = exp._wrap_model(model)
        self.assertEqual(wrapped[2], [0, 1])

    def test_ddp_uses_local_rank(self):
        exp = self._bare_exp(use_gpu=True, use_ddp=1)
        exp.local_rank = 1
        model = nn.Linear(2, 1)
        with mock.patch("utils.ddp_utils.is_dist", return_value=True), \
             mock.patch("pre_training.experiments.exp_basic.nn.parallel.DistributedDataParallel") as ddp_mock:
            ddp_mock.side_effect = lambda m, device_ids: ("ddp", m, device_ids)
            wrapped = exp._wrap_model(model)
        self.assertEqual(wrapped[2], [1])

    def test_ddp_skip_when_not_initialized(self):
        """单进程误开 use_ddp 且进程组未初始化时跳过包装而不是报错。"""
        exp = self._bare_exp(use_gpu=True, use_ddp=1)
        model = nn.Linear(2, 1)
        self.assertIs(exp._wrap_model(model), model)

    def test_no_wrap_without_gpu(self):
        exp = self._bare_exp(use_gpu=False, use_dp=1, use_ddp=1)
        model = nn.Linear(2, 1)
        self.assertIs(exp._wrap_model(model), model)


class TestEntryBranches(unittest.TestCase):
    """run_gpt2_124M 根入口：inference/test 分支与 main() 的 DDP/seed 决策。"""

    def _args(self, **overrides):
        args = SimpleNamespace(
            task_name="t", model_name="gpt2_124M", data_name="d", context_length=8,
            train_epochs=1, batch_size=1, itrs=1,
            is_train=0, is_test=0, is_inference=0, use_ddp=0, seed=42,
        )
        for k, v in overrides.items():
            setattr(args, k, v)
        return args

    def test_inference_branch_builds_exp(self):
        """is_train=0 + is_inference=1 时 exp 必须新建且按关键字调用（修复前 exp 未定义）。"""
        import run_gpt2_124M
        calls = []

        class _FakeExp:
            def __init__(self, args, global_rank, world_size, local_rank):
                pass

            def inference(self, epoch, setting, load=False, **kwargs):
                calls.append(("inference", epoch, setting, load))
                return "some completion"

        with mock.patch.object(run_gpt2_124M, "Model_Pretrain", _FakeExp):
            run_gpt2_124M.run(self._args(is_inference=1), 0, 1, 0)
        self.assertEqual(len(calls), 1)
        _, epoch, _, load = calls[0]
        self.assertEqual(epoch, 0)
        self.assertTrue(load)

    def test_test_branch_builds_exp(self):
        """is_train=0 + is_test=1 时 exp 必须新建并调用 test(setting)。"""
        import run_gpt2_124M
        calls = []

        class _FakeExp:
            def __init__(self, args, global_rank, world_size, local_rank):
                pass

            def test(self, setting):
                calls.append(("test", setting))
                return 1.0

        with mock.patch.object(run_gpt2_124M, "Model_Pretrain", _FakeExp):
            run_gpt2_124M.run(self._args(is_test=1), 0, 1, 0)
        self.assertEqual(len(calls), 1)

    def test_main_uses_args_seed_and_no_dist_by_default(self):
        """use_ddp=0 时不初始化进程组；seed 来自 args.seed（修复前硬编码 2025）。"""
        import run_gpt2_124M
        with mock.patch.object(run_gpt2_124M, "init_dist") as init_mock, \
             mock.patch.object(run_gpt2_124M, "set_seed_torch") as seed_mock, \
             mock.patch.object(run_gpt2_124M, "print_master_log"), \
             mock.patch.object(run_gpt2_124M, "run"):
            run_gpt2_124M.main(self._args(seed=7), 0, 1, 0)
        init_mock.assert_not_called()
        seed_mock.assert_called_once_with(seed=7)

    def test_main_inits_dist_when_ddp(self):
        """use_ddp=1 且 WORLD_SIZE>1 时初始化进程组并传入 local_rank。"""
        import run_gpt2_124M
        with mock.patch.object(run_gpt2_124M, "init_dist") as init_mock, \
             mock.patch.object(run_gpt2_124M, "destory_dist") as destroy_mock, \
             mock.patch.object(run_gpt2_124M, "set_seed_torch") as seed_mock, \
             mock.patch.object(run_gpt2_124M, "print_master_log"), \
             mock.patch("torch.distributed.barrier") as barrier_mock, \
             mock.patch.object(run_gpt2_124M, "run"):
            run_gpt2_124M.main(self._args(use_ddp=1, seed=7), 1, 2, 1)
        init_mock.assert_called_once_with(1, 2, 1)
        # DDP 下各 rank 的 seed 加偏移
        seed_mock.assert_called_once_with(seed=8)
        destroy_mock.assert_called_once()

    def test_qkv_bias_flag_semantics(self):
        """--qkv_bias 默认 True 且传入 flag 后仍为 True（修复前 store_false 会关掉）。"""
        import run_gpt2_124M
        parser = run_gpt2_124M.build_parser()
        self.assertTrue(parser.parse_args([]).qkv_bias)
        self.assertTrue(parser.parse_args(["--qkv_bias"]).qkv_bias)


class TestCheckpointResume(unittest.TestCase):
    """checkpoint 保存/恢复闭环（model + optimizer + step + extra）。"""

    def test_save_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = CheckpointManager(tmp, keep_last=2)
            torch.manual_seed(0)
            model = nn.Linear(4, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            # 走一步，让 optimizer 有状态
            loss = model(torch.randn(3, 4)).sum()
            loss.backward()
            optimizer.step()
            manager.save(step=5, model=model, optimizer=optimizer,
                         extra={"epoch": 1, "batch": 3, "global_step": 5})
            # 新实例恢复
            model2 = nn.Linear(4, 2)
            optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
            payload = manager.load_latest(model=model2, optimizer=optimizer2)
            self.assertEqual(payload["step"], 5)
            self.assertEqual(payload["extra"]["epoch"], 1)
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                self.assertTrue(torch.equal(p1, p2))

    def test_rotation_keeps_last_n(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = CheckpointManager(tmp, keep_last=2)
            for step in range(4):
                manager.save(step=step, model=nn.Linear(2, 1))
            self.assertEqual(len(list(Path(tmp).glob("step_*.pt"))), 2)


class TestUnwrapModel(unittest.TestCase):
    def test_unwrap_dp_module(self):
        model = nn.Linear(2, 1)
        wrapped = torch.nn.DataParallel(model, device_ids=[])
        self.assertIs(Model_Pretrain._unwrap_model(wrapped), model)
        self.assertIs(Model_Pretrain._unwrap_model(model), model)


class TestLegacyEntries(unittest.TestCase):
    """其他模型的根目录预训练入口可导入（ch05/14）。"""

    def test_import_legacy_entries(self):
        for mod in ("run_llama", "run_qwen"):
            self.assertIsNotNone(importlib.import_module(mod))


if __name__ == "__main__":
    unittest.main()
