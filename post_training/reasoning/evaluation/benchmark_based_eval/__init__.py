import importlib
import sys

for _name in ("math_verifiers", "multiple_choice"):
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(f"inference.evaluation.benchmarks.{_name}")
