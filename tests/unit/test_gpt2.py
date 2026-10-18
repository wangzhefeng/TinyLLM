import unittest
from types import SimpleNamespace

import torch

from models.gpt2.gpt2_124M import Model
from utils.calc_loss import calc_loss_batch
from utils.optimizer import select_optimizer


class TestGPT2(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.cfg = SimpleNamespace(
            vocab_size=128,
            context_length=16,
            embed_dim=32,
            d_ff=64,
            n_heads=4,
            n_layers=2,
            dropout=0.0,
            qkv_bias=True,
            dtype=torch.float32,
            is_train=False,
        )
        self.model = Model(self.cfg)

    def test_forward_shape(self):
        x = torch.randint(0, self.cfg.vocab_size, (2, 8))
        logits = self.model(x)
        self.assertEqual(logits.shape, (2, 8, self.cfg.vocab_size))

    def test_loss_backward_and_step(self):
        x = torch.randint(0, self.cfg.vocab_size, (2, 8))
        y = torch.randint(0, self.cfg.vocab_size, (2, 8))
        optimizer = select_optimizer(self.model, learning_rate=1e-4, weight_decay=0.0)
        loss = calc_loss_batch(
            "pretrain",
            x,
            y,
            self.model,
            torch.nn.CrossEntropyLoss(),
            torch.device("cpu"),
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        optimizer.step()
        self.assertGreater(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
