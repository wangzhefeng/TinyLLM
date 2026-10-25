import copy
import unittest

from utils.args_tools import DotDict


class TestDotDict(unittest.TestCase):
    def setUp(self):
        self.dd = DotDict({"a": 1, "nested": {"b": 2}})

    def test_attribute_access(self):
        self.assertEqual(self.dd.a, 1)
        self.assertEqual(self.dd.nested.b, 2)

    def test_getattr_returns_default_for_missing_key(self):
        self.assertIsNone(getattr(self.dd, "dtype", None))
        self.assertEqual(getattr(self.dd, "missing", "fallback"), "fallback")

    def test_missing_attribute_raises_attribute_error(self):
        with self.assertRaises(AttributeError):
            self.dd.missing
        self.assertFalse(hasattr(self.dd, "missing"))

    def test_missing_item_still_raises_key_error(self):
        with self.assertRaises(KeyError):
            self.dd["missing"]

    def test_set_and_delete_attribute(self):
        self.dd.c = 3
        self.assertEqual(self.dd["c"], 3)
        del self.dd.c
        self.assertNotIn("c", self.dd)

    def test_deepcopy_preserves_contract(self):
        cloned = copy.deepcopy(self.dd)
        self.assertEqual(cloned.a, 1)
        self.assertEqual(cloned.nested.b, 2)
        self.assertIsNone(getattr(cloned, "dtype", None))


if __name__ == "__main__":
    unittest.main()
