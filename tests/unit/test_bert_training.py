import random
import unittest
from argparse import Namespace
from pathlib import Path
import tempfile

import torch

from models.bert.model import Model
from pre_training.data_provider.bert.data_loader import BertDataset
from run_bert import (
    build_config,
    build_dataloaders,
    evaluate,
    save_checkpoint,
)


def _toy_args(**overrides):
    args = Namespace(
        dev_size=4,
        max_len=128,
        batch_size=4,
        num_workers=0,
        hidden_size=32,
        feed_num=64,
        head_num=4,
        layer_num=2,
        type_vocab_size=3,
        hidden_dropout_prob=0.0,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def _toy_data(n=20):
    word_2_index = {"[PAD]": 0, "[unused1]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4, "[UNK]": 5}
    for w in "今天天气很好我们去公园散步明天会更好": 
        if w not in word_2_index:
            word_2_index[w] = len(word_2_index)
    texts = ["今天天气很好", "我们去公园散步", "明天会更好", "今天去公园"]
    all_text1 = [texts[i % len(texts)] for i in range(n)]
    all_text2 = [texts[(i + 1) % len(texts)] for i in range(n)]
    all_label = [i % 2 for i in range(n)]
    return all_text1, all_text2, all_label, word_2_index


class TestBertTrainingPipeline(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.args = _toy_args()
        self.config = build_config(self.args, vocab_size=31)
        self.model = Model(self.config)
        data = _toy_data()
        self.train_loader, self.dev_loader = build_dataloaders(data[0], data[1], data[2], data[3], self.args)

    def test_train_step_and_evaluate(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        batch_idx, batch_label, batch_mask_val, batch_seg_idx = next(iter(self.train_loader))
        self.assertEqual(batch_idx.shape[1], self.args.max_len)
        loss = self.model(batch_idx, batch_seg_idx, batch_mask_val, batch_label)
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        val_loss, acc_mask, acc_next = evaluate(self.model, self.dev_loader, torch.device("cpu"))
        self.assertTrue(torch.isfinite(torch.tensor(val_loss)))
        self.assertGreaterEqual(acc_mask, 0.0)
        self.assertGreaterEqual(acc_next, 0.0)

    def test_checkpoint_save_and_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = str(Path(tmpdir) / "sub" / "ckpt.pt")
            save_checkpoint(self.model, None, epoch=0, val_loss=1.0, config=self.config, ckpt_path=ckpt)
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            self.assertEqual(state["config"], self.config)
            new_model = Model(state["config"])
            new_model.load_state_dict(state["model_state_dict"])
            x = torch.randint(0, 31, (2, 8))
            seg = torch.zeros_like(x)
            self.model.eval()
            new_model.eval()
            out1 = self.model(x, seg)[1]
            out2 = new_model(x, seg)[1]
            self.assertTrue(torch.equal(out1, out2))


class TestBertMlmMasking(unittest.TestCase):
    """MLM 80/10/10：选中 token 的 10% 保持不变时也必须保留预测标签。"""

    def test_unchanged_branch_keeps_label(self):
        _, _, _, word_2_index = _toy_data()
        ds = BertDataset(["今天天气很好"], ["我们去公园散步"], [1], 128, word_2_index)
        mask_id = word_2_index["[MASK]"]
        saw_unchanged, saw_mask, saw_random = False, False, False
        random.seed(2025)
        for _ in range(300):
            text_idx, _, mask_val, _ = ds[0]
            for tok, label in zip(text_idx.tolist(), mask_val.tolist()):
                if label == 0:
                    continue  # 非预测位置
                if tok == mask_id:
                    saw_mask = True
                elif tok == label:
                    # token 未变但仍是被预测目标（修复前该分支不写标签，永远走不到这里）
                    saw_unchanged = True
                else:
                    saw_random = True
        self.assertTrue(saw_mask, "80% [MASK] 分支未出现")
        self.assertTrue(saw_unchanged, "10% 保持不变分支未保留标签")
        self.assertTrue(saw_random, "10% 随机替换分支未出现")


class TestEvaluateSingleForward(unittest.TestCase):
    """evaluate 每个 batch 只做一次前向（修复前 loss 与 accuracy 各一次）。"""

    def test_single_forward_per_batch(self):
        torch.manual_seed(123)
        args = _toy_args()
        config = build_config(args, vocab_size=31)

        class _CountingModel(Model):
            n_forward = 0

            def forward(self, *a, **k):
                type(self).n_forward += 1
                return super().forward(*a, **k)

        model = _CountingModel(config)
        data = _toy_data()
        _, dev_loader = build_dataloaders(data[0], data[1], data[2], data[3], args)
        val_loss, acc_mask, acc_next = evaluate(model, dev_loader, torch.device("cpu"))
        self.assertEqual(_CountingModel.n_forward, len(dev_loader))
        self.assertTrue(torch.isfinite(torch.tensor(val_loss)))
        self.assertGreaterEqual(acc_mask, 0.0)
        self.assertGreaterEqual(acc_next, 0.0)

    def test_return_preds_matches_two_forward(self):
        """return_preds=True 的预测结果与无标签前向一致。"""
        torch.manual_seed(123)
        args = _toy_args()
        config = build_config(args, vocab_size=31)
        model = Model(config)
        model.eval()
        data = _toy_data()
        _, dev_loader = build_dataloaders(data[0], data[1], data[2], data[3], args)
        batch_idx, batch_label, batch_mask_val, batch_seg_idx = next(iter(dev_loader))
        with torch.no_grad():
            loss, pre_mask, pre_next = model(
                batch_idx, batch_seg_idx, batch_mask_val, batch_label, return_preds=True
            )
            pre_mask2, pre_next2 = model(batch_idx, batch_seg_idx)
        self.assertTrue(torch.equal(pre_mask, pre_mask2))
        self.assertTrue(torch.equal(pre_next, pre_next2))
        self.assertTrue(torch.isfinite(loss))



if __name__ == "__main__":
    unittest.main()
