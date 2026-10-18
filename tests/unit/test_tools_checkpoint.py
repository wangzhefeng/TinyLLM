import tempfile
import unittest
from pathlib import Path

import torch

from tools.checkpoint import (
    CheckpointManager,
    latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)


class CheckpointTest(unittest.TestCase):
    def test_save_load_roundtrip(self):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        expected = {k: v.clone() for k, v in model.state_dict().items()}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                step=7,
                extra={"note": "demo"},
            )
            self.assertTrue(path.exists())

            restored = torch.nn.Linear(2, 2)
            payload = load_checkpoint(path, model=restored)
            self.assertEqual(payload["step"], 7)
            self.assertEqual(payload["extra"]["note"], "demo")
            for key, value in restored.state_dict().items():
                self.assertTrue(torch.allclose(value, expected[key]))

    def test_manager_rotation_and_latest(self):
        model = torch.nn.Linear(1, 1)
        with tempfile.TemporaryDirectory() as tmp:
            manager = CheckpointManager(tmp, keep_last=2)
            for step in range(1, 5):
                manager.save(step, model=model)
            files = sorted(p.name for p in Path(tmp).glob("step_*.pt"))
            self.assertEqual(files, ["step_00000003.pt", "step_00000004.pt"])
            self.assertIsNotNone(latest_checkpoint(tmp))
            payload = manager.load_latest(model=model)
            self.assertEqual(payload["step"], 4)


if __name__ == "__main__":
    unittest.main()
