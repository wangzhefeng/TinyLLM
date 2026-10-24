import tempfile
import unittest
from pathlib import Path

from layers.tokenizers.tokenization import text_to_token_ids, token_ids_to_text
from layers.tokenizers.unigram.unigram import UnigramTokenizer


class TestTokenizer(unittest.TestCase):
    def test_gpt2_roundtrip(self):
        try:
            ids = text_to_token_ids("Hello, TinyLLM.")
        except Exception as exc:  # tiktoken may need the GPT-2 vocab on first use
            self.skipTest(f"tokenizer unavailable: {exc}")
        self.assertEqual(ids.ndim, 2)
        self.assertGreater(ids.shape[1], 0)
        text = token_ids_to_text(ids)
        self.assertIn("TinyLLM", text)


class TestUnigramTokenizer(unittest.TestCase):
    CORPUS = [
        "the quick brown fox jumps over the lazy dog",
        "the dog barks at the fox",
        "a quick brown dog runs fast",
        "the lazy dog sleeps all day",
        "foxes are quick and clever animals",
        "the brown fox is quick",
        "dogs and foxes are animals",
        "the cat sleeps on the mat",
        "a lazy cat sleeps all day",
        "the quick dog jumps over the lazy cat",
        "she sells sea shells by the sea shore",
        "the sea is blue and deep",
        "a fast fox runs in the forest",
        "the forest is dark and deep",
        "brown dogs and black cats are friends",
        "the sun rises in the east",
        "a red fox runs fast in the sun",
        "lazy dogs sleep in the warm sun",
        "the clever fox outruns the slow dog",
        "cats and dogs are common pets",
    ]
    VOCAB_SIZE = 300

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = UnigramTokenizer(vocab_size=cls.VOCAB_SIZE)
        cls.tokenizer.train(cls.CORPUS)

    @staticmethod
    def _norm_ws(text):
        return " ".join(text.split())

    def test_encode_decode_roundtrip(self):
        for text in self.CORPUS:
            ids = self.tokenizer.encode(text)
            self.assertTrue(ids)
            self.assertEqual(self._norm_ws(self.tokenizer.decode(ids)), self._norm_ws(text))

    def test_unseen_chars_fallback_to_unk(self):
        # 语料中没有数字与感叹号：应走 unk，不抛异常
        ids = self.tokenizer.encode("the fox 123!!!")
        self.assertTrue(ids)
        self.assertIn(self.tokenizer.unk_id, ids)

    def test_save_load_roundtrip(self):
        text = "the quick brown fox"
        ids_before = self.tokenizer.encode(text)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "unigram.json"
            self.tokenizer.save(path)
            loaded = UnigramTokenizer.load(path)
        self.assertEqual(loaded.encode(text), ids_before)
        self.assertEqual(loaded.n_vocab, self.tokenizer.n_vocab)

    def test_vocab_size_and_base_chars(self):
        self.assertLessEqual(self.tokenizer.n_vocab, self.VOCAB_SIZE)
        base_chars = {ch for s in self.CORPUS for ch in s}
        for ch in base_chars:
            self.assertIn(ch, self.tokenizer.vocab)
        self.assertIn(self.tokenizer.unk_token, self.tokenizer.vocab)


if __name__ == "__main__":
    unittest.main()
