import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from models.gpt2.gpt2_124M import Model


def model_config():
    return SimpleNamespace(
        vocab_size=50304, context_length=16, embed_dim=16, d_ff=32,
        n_heads=4, n_layers=1, dropout=0.0, qkv_bias=True,
        dtype=torch.float32,
    )


def batches():
    generator = torch.Generator().manual_seed(0)
    inputs = torch.randint(0, 100, (4, 8), generator=generator)
    targets = torch.randint(0, 100, (4, 8), generator=generator)
    return DataLoader(TensorDataset(inputs, targets), batch_size=2)


class TestTinyPostTraining(unittest.TestCase):
    def test_instruction_sft_train_and_save(self):
        from post_training.sft.exp.exp_finetune_gpt_instruction_flow import ModelFinetuningInstructionFlow

        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                task_name="tiny_gpt_instruction_sft", tokenizer_model="gpt2",
                pretrained_model_source="local", pretrained_model="gpt2-small (124M)",
                learning_rate=1e-3, weight_decay=0.0, train_epochs=1,
                finetuned_model_path=str(Path(tmp) / "models"),
                test_results=str(Path(tmp) / "results"), max_new_tokens=1,
                context_length=16, is_eval=0, seed=42,
            )
            exp = ModelFinetuningInstructionFlow(args)
            exp.device = torch.device("cpu")
            entry = {"instruction": "Repeat", "input": "hello", "output": "hello"}
            data = ([entry], None, batches(), [entry], None, batches(), [entry], None, batches())
            with patch.object(exp, "_get_data", return_value=data), patch(
                "post_training.sft.exp.exp_finetune_gpt_instruction_flow.model_with_gpt2_weights",
                return_value=(Model(model_config()), model_config()),
            ):
                exp.train(training_iter=0, setting="tiny_sft", eval_freq=1, eval_iter=1)
            self.assertTrue(Path(exp.model_path).is_file())
            restored = Model(model_config())
            exp._load_model(restored)

    def test_dpo_train_and_save(self):
        from post_training.preference.exp.exp_finetune_gpt_dpo import ModelFinetuningPreference

        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                tokenizer_model="gpt2", pretrained_model="gpt2-medium (355M)",
                learning_rate=1e-3, weight_decay=0.0, train_epochs=1,
                model_path=str(Path(tmp) / "models"),
                test_results=str(Path(tmp) / "results"),
                finetuned_model_path="unused.pth", beta=0.1,
                dpo_avg_logprobs=0,
            )
            exp = ModelFinetuningPreference(args)
            exp.device = torch.device("cpu")
            chosen = torch.randint(0, 100, (2, 8))
            rejected = torch.randint(0, 100, (2, 8))
            batch = {
                "chosen": chosen,
                "rejected": rejected,
                "chosen_mask": torch.ones_like(chosen),
                "rejected_mask": torch.ones_like(rejected),
            }
            data = ([], None, [batch], [], None, [batch], [], None, [batch])
            with patch.object(exp, "_get_data", return_value=data), patch.object(
                exp,
                "_build_policy_reference_model",
                return_value=(Model(model_config()), Model(model_config())),
            ):
                exp.train(training_iter=0, setting="tiny_dpo", eval_freq=1, eval_iter=1)
            self.assertTrue(Path(exp.model_path).is_file())
            self.assertTrue(str(exp.model_path).endswith("-dpo.pth"))


if __name__ == "__main__":
    unittest.main()
