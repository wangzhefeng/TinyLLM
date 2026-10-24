import importlib
import sys

_modules = ("load_gpt2_pretrained_weights", "load_pretrained_weights", "openai_gpt2_weights_load", "openai_gpt2_weights_load_hf", "openai_gpt2_weights_load_hf_safetensors")
for _name in _modules:
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(f"models.io.gpt2.{_name}")
