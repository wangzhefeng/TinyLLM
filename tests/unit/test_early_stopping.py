# -*- coding: utf-8 -*-
"""
EarlyStopping 单元测试：

- checkpoint 的 epoch 按传入值（0-based）原样保存，不再 +1
- optimizer state 新写 "optimizer"，不再写历史笔误键
- load_training_state 兼容读取只含 "optmizer" 的旧 checkpoint
- patience 计数与 early_stop 触发
"""

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from utils.early_stopping import EarlyStopping, load_training_state


class TestEarlyStoppingCheckpoint(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.model = nn.Linear(2, 1)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

    def test_epoch_saved_without_increment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "ckpt.pth")
            es = EarlyStopping(patience=2, verbose=False)
            es(epoch=0, val_loss=1.0, model=self.model, optimizer=self.optimizer, model_path=path)
            state = torch.load(path, map_location="cpu", weights_only=False)
            # 传入 0-based epoch 0，保存的也必须是 0（修复前会被 +1）
            self.assertEqual(state["epoch"], 0)

    def test_optimizer_uses_canonical_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "ckpt.pth")
            es = EarlyStopping(patience=2, verbose=False)
            es(epoch=0, val_loss=1.0, model=self.model, optimizer=self.optimizer, model_path=path)
            state = torch.load(path, map_location="cpu", weights_only=False)
            self.assertIn("optimizer", state)
            self.assertNotIn("optmizer", state)
            self.assertIsNotNone(state["optimizer"])
            self.assertIn("model", state)

    def test_load_training_state_reads_legacy_typo_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "legacy.pth")
            # 模拟只含历史笔误键的旧 checkpoint
            torch.save({
                "epoch": 3,
                "model": self.model.state_dict(),
                "optmizer": self.optimizer.state_dict(),
                "scheduler": None,
            }, path)
            state = load_training_state(path, map_location="cpu")
            self.assertIn("optimizer", state)
            self.assertEqual(state["epoch"], 3)

    def test_counter_and_early_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "ckpt.pth")
            es = EarlyStopping(patience=2, verbose=False)
            es(epoch=0, val_loss=1.0, model=self.model, optimizer=self.optimizer, model_path=path)
            self.assertFalse(es.early_stop)
            es(epoch=1, val_loss=1.1, model=self.model, optimizer=self.optimizer, model_path=path)
            self.assertEqual(es.counter, 1)
            self.assertFalse(es.early_stop)
            es(epoch=2, val_loss=1.2, model=self.model, optimizer=self.optimizer, model_path=path)
            self.assertTrue(es.early_stop)
            # 最优 loss 仍未被覆盖
            self.assertAlmostEqual(es.val_loss_min, 1.0)

    def test_improvement_resets_counter(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "ckpt.pth")
            es = EarlyStopping(patience=2, verbose=False)
            es(epoch=0, val_loss=1.0, model=self.model, model_path=path)
            es(epoch=1, val_loss=1.1, model=self.model, model_path=path)
            self.assertEqual(es.counter, 1)
            es(epoch=2, val_loss=0.9, model=self.model, model_path=path)
            self.assertEqual(es.counter, 0)
            state = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(state["epoch"], 2)


if __name__ == "__main__":
    unittest.main()
