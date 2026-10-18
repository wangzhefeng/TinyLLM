import unittest

import torch

from models.bert.model import Model


class TestBert(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.config = {
            "vocab_size": 100,
            "hidden_size": 32,
            "feed_num": 64,
            "head_num": 4,
            "layer_num": 2,
            "max_position_embeddings": 16,
            "type_vocab_size": 2,
            "hidden_dropout_prob": 0.0,
        }
        self.model = Model(self.config)

    def test_forward_returns_mlm_and_nsp_predictions(self):
        batch_idx = torch.randint(0, self.config["vocab_size"], (2, 8))
        batch_seg_idx = torch.zeros_like(batch_idx)
        mlm_pred, nsp_pred = self.model(batch_idx, batch_seg_idx)
        self.assertEqual(mlm_pred.shape, (2, 8))
        self.assertEqual(nsp_pred.shape, (2,))

    def test_loss_backward(self):
        batch_idx = torch.randint(0, self.config["vocab_size"], (2, 8))
        batch_seg_idx = torch.zeros_like(batch_idx)
        mask_labels = torch.randint(0, self.config["vocab_size"], (2, 8))
        next_labels = torch.randint(0, 2, (2,))
        loss = self.model(batch_idx, batch_seg_idx, mask_labels, next_labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()


if __name__ == "__main__":
    unittest.main()
