import unittest

import torch
from torch.utils.data import DataLoader

from post_training.sft.bert_sentiment.data import SentimentDataset
from post_training.sft.bert_sentiment.model import BertSentimentClassifier
from post_training.sft.bert_sentiment.run import evaluate


def _toy_vocab():
    w2i = {"[PAD]": 0, "[unused1]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4, "[UNK]": 5}
    for w in "好书太差喜欢了":
        if w not in w2i:
            w2i[w] = len(w2i)
    return w2i


def _toy_config(vocab_size):
    return {
        "vocab_size": vocab_size,
        "hidden_size": 32,
        "feed_num": 64,
        "head_num": 4,
        "layer_num": 2,
        "max_position_embeddings": 32,
        "type_vocab_size": 3,
        "hidden_dropout_prob": 0.0,
    }


class TestBertSentiment(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.w2i = _toy_vocab()
        texts = ["好书喜欢", "太差了", "喜欢好书", "差了太差"] * 4
        labels = [1, 0, 1, 0] * 4
        self.loader = DataLoader(
            SentimentDataset(texts, labels, self.w2i, max_len=32),
            batch_size=4, shuffle=False,
        )

    def test_dataset_item_shapes(self):
        input_ids, seg_ids, label = next(iter(self.loader))
        self.assertEqual(input_ids.shape, (4, 32))
        self.assertEqual(seg_ids.shape, (4, 32))
        self.assertEqual(label.shape, (4,))

    def test_train_step_and_evaluate(self):
        model = BertSentimentClassifier(_toy_config(len(self.w2i)), init_mode="scratch")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        input_ids, seg_ids, labels = next(iter(self.loader))
        loss = model(input_ids, seg_ids, labels)
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        _, acc = evaluate(model, self.loader, torch.device("cpu"))
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_freeze_backbone(self):
        model = BertSentimentClassifier(
            _toy_config(len(self.w2i)), init_mode="scratch", freeze_backbone=True
        )
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        self.assertTrue(all(n.startswith("classifier") for n in trainable))


if __name__ == "__main__":
    unittest.main()
