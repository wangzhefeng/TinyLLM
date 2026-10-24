import importlib
import sys

sys.modules[f"{__name__}.qwen3_06B"] = importlib.import_module("inference.strategies.llm_basic.qwen3_06B")
