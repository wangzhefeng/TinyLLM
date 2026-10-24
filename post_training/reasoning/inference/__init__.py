import importlib
import sys

for _name in ("advance_text_generation", "voting_and_verification"):
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(f"inference.strategies.{_name}")
