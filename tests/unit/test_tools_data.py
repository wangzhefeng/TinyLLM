import tempfile
import unittest
from pathlib import Path

from tools.data import (
    dedupe_jsonl,
    dedupe_records,
    normalize_text,
    pack_token_ids,
    read_jsonl,
    write_jsonl,
)


class DataToolsTest(unittest.TestCase):
    def test_normalize_and_dedupe(self):
        self.assertEqual(normalize_text("  A   b\n", lowercase=True), "a b")
        records = [
            {"text": "Hello   world"},
            {"text": "hello world"},
            {"text": ""},
            {"text": "other"},
        ]
        kept = list(dedupe_records(records))
        self.assertEqual([r["text"] for r in kept], ["Hello   world", "other"])

    def test_jsonl_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            written = write_jsonl([{"text": "a"}, {"text": "b"}], path)
            self.assertEqual(written, 2)
            self.assertEqual(read_jsonl(path), [{"text": "a"}, {"text": "b"}])

    def test_dedupe_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.jsonl"
            dst = Path(tmp) / "out.jsonl"
            write_jsonl([{"text": "x"}, {"text": "x "}, {"text": "y"}], src)
            self.assertEqual(dedupe_jsonl(src, dst), 2)
            self.assertEqual(len(read_jsonl(dst)), 2)

    def test_pack_token_ids(self):
        self.assertEqual(
            list(pack_token_ids([1, 2, 3, 4, 5], max_length=2)),
            [[1, 2], [3, 4], [5]],
        )
        self.assertEqual(
            list(pack_token_ids([1, 2, 3], max_length=2, pad_id=0)),
            [[1, 2], [3, 0]],
        )
        self.assertEqual(
            list(pack_token_ids([1, 2, 3], max_length=2, drop_last=True)),
            [[1, 2]],
        )


if __name__ == "__main__":
    unittest.main()
