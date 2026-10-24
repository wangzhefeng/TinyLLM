import importlib
import sys

_modules = ("llm_judges", "llm_judges_Ollama_API", "llm_judges_Qwen3", "llm_leaderboards", "elo_leaderboard", "bradley_terry_leaderboard")
for _name in _modules:
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(f"inference.evaluation.judges.{_name}")
from inference.evaluation.judges.llm_leaderboards import *
