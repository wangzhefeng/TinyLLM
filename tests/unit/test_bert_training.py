import unittest
from argparse import Namespace
from pathlib import Path
import tempfile

import torch

from models.bert.model import Model
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


if __name__ == "__main__":
    unittest.main()
