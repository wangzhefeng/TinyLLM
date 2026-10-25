import argparse
import sys
from pathlib import Path

ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)


TASKS = {
    "sft_clf": "post_training.sft.run.run_gpt_clf_sft",
    "sft_ins": "post_training.sft.run.run_gpt_instruction_sft",
    "dpo": "post_training.preference.model_rlhf.dpo.run_gpt_dpo",
    "grpo": "post_training.rl.run",
    "reinforce": "post_training.rl.run",
    "distill_reasoning": "post_training.reasoning.run",
}


def main():
    parser = argparse.ArgumentParser(description="TinyLLM post-training entry point")
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    args, remaining = parser.parse_known_args()
    from importlib import import_module
    from utils.random_seed import set_seed_torch

    entry = import_module(TASKS[args.task])
    sys.argv = [f"post_training/run.py --task {args.task}", *remaining]
    task_args = entry.args_parse()
    set_seed_torch(task_args.seed)
    entry.run(task_args)


if __name__ == "__main__":
    main()
