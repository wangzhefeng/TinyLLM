import importlib
import sys

for _name in ("load_weights", "model_download", "qwen3_06B_load", "qwen3_30B_A3B_moe_load"):
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(f"models.io.qwen3.{_name}")
