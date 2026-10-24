import importlib
import sys

for _name in ("load_llama2_pretrained_weights", "load_llama3_pretrained_weights", "meta_llama2_weights_load_hf", "meta_llama3_weights_load_hf"):
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(f"models.io.llama.{_name}")
