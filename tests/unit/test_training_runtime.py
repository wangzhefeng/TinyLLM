import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from training.checkpoint import load_training_checkpoint, save_training_checkpoint
from training.trainer import Trainer


class TestTrainer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = torch.nn.Linear(2, 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

    def test_accumulation_and_optimizer_steps(self):
        trainer = Trainer(self.model, self.optimizer, "cpu", grad_accum_steps=2)
        before = self.model.weight.detach().clone()
        batches = [
            (torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0]])),
            (torch.tensor([[0.0, 1.0]]), torch.tensor([[2.0]])),
        ]

        def loss_fn(batch):
            x, y = batch
            return torch.nn.functional.mse_loss(self.model(x), y)

        _, first_step = trainer.train_step(batches[0], loss_fn, 0, 2)
        _, second_step = trainer.train_step(batches[1], loss_fn, 1, 2)
        self.assertFalse(first_step)
        self.assertTrue(second_step)
        self.assertFalse(torch.equal(before, self.model.weight))

    def test_amp_is_disabled_off_cuda_and_eval_restores_mode(self):
        trainer = Trainer(self.model, self.optimizer, "cpu", use_amp=True)
        self.assertFalse(trainer.use_amp)
        self.model.train()

        def loss_fn(batch):
            x, y = batch
            return torch.nn.functional.mse_loss(self.model(x), y)

        loader = [(torch.ones(1, 2), torch.zeros(1, 1))]
        self.assertGreater(trainer.evaluate(loader, loss_fn), 0.0)
        self.assertTrue(self.model.training)


class TestCheckpointContract(unittest.TestCase):
    def test_canonical_roundtrip_and_legacy_optimizer(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            save_training_checkpoint(path, model, optimizer, epoch=2, step=7, config={"name": "tiny"})
            payload = load_training_checkpoint(path, model, optimizer)
            self.assertEqual(payload["epoch"], 2)
            self.assertEqual(payload["step"], 7)
            self.assertEqual(payload["config"], {"name": "tiny"})
            self.assertIn("model_state_dict", payload)
            self.assertIn("optimizer_state_dict", payload)

    def test_post_training_unified_cli(self):
        for task in ("sft_clf", "sft_ins", "dpo"):
            result = subprocess.run(
                [sys.executable, "post_training/run.py", "--task", task, "--help"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)


if __name__ == "__main__":
    unittest.main()
