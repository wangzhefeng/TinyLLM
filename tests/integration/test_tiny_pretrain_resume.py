import tempfile
import unittest
from pathlib import Path

from pre_training.experiments.exp_pretrain_gpt2_124M import Model_Pretrain
from run_gpt2_124M import build_parser, finalize_args


class TestTinyPretrainResume(unittest.TestCase):
    def test_train_checkpoint_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tiny.txt").write_text(
                "The quick brown fox jumps over the lazy dog. " * 80,
                encoding="utf-8",
            )
            args = finalize_args(build_parser().parse_args([
                "--data_path", str(root),
                "--data_file", "tiny.txt",
                "--context_length", "16",
                "--embed_dim", "16",
                "--d_ff", "32",
                "--n_heads", "4",
                "--n_layers", "2",
                "--vocab_size", "50304",
                "--batch_size", "2",
                "--train_epochs", "1",
                "--train_ratio", "0.8",
                "--use_gpu", "0",
                "--use_amp", "0",
                "--dtype", "float32",
                "--max_new_tokens", "2",
                "--save_every_steps", "1",
                "--keep_last_checkpoints", "1",
                "--checkpoints", str(root / "checkpoints"),
                "--test_results", str(root / "results"),
            ]))
            exp = Model_Pretrain(args, 0, 1, 0)
            exp.train(training_iter=0, setting="tiny", eval_freq=1, eval_iter=1)
            checkpoint = root / "checkpoints" / "tiny" / "checkpoint.pth"
            self.assertTrue(checkpoint.is_file())
            self.assertTrue(list((root / "checkpoints" / "tiny" / "steps").glob("step_*.pt")))

            args.train_epochs = 2
            args.resume = "latest"
            resumed = Model_Pretrain(args, 0, 1, 0)
            resumed.train(training_iter=0, setting="tiny", eval_freq=1, eval_iter=1)
            self.assertTrue(checkpoint.is_file())


if __name__ == "__main__":
    unittest.main()
