import argparse
import sys
from pathlib import Path

ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)


TASKS = {
    "sft_clf": "post_training.sft.run.run_gpt_clf_sft",
    "sft_ins": "post_training.sft.run.run_gpt_instruction_sft",
    "dpo": "post_training.preference.run.run_gpt_dpo",
    "grpo": "post_training.rl.run",
    "reinforce": "post_training.rl.run",
    "distill_reasoning": "post_training.reasoning.run",
}

# 未纳入分发器的入口，手动运行：
#   - PPO（序列级 bandit 教学实现）: .venv/bin/python post_training/rl/ppo.py
#   - RM 训练: .venv/bin/python post_training/preference/exp/exp_train_rm.py
#   - peft 演示（lora_demo / prompt_tuning / scratch_exps）: post_training/peft/demos/
#   - post_training/references/ 为外部栈（TRL / unsloth / HF PEFT）对照实验，不接入分发器。


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
