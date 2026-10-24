import importlib
import sys

sys.modules[f"{__name__}.load_weights"] = importlib.import_module("models.io.gemma3.load_weights")
