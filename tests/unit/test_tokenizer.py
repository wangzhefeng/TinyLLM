import unittest

from layers.tokenizers.tokenization import text_to_token_ids, token_ids_to_text


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


if __name__ == "__main__":
    unittest.main()
