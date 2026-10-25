> 所属：[TinyLLM Docs](../README.md) › 评测公共库

# 评测公共库（evaluation）

`evaluation/` 是**三域共用**的评测基础设施层：外部 LLM 客户端、数学答案验证器、LLM-as-a-Judge 与排行榜。推断 harness、后训练 SFT/DPO/RL 实验、文档示例都从它 import，避免各域各抄一份客户端与验证器。包级 `__init__.py` 懒加载导出常用函数，import 本包不拉起 torch。

## 目录结构

```text
evaluation/
├── clients/          # 外部 LLM 服务客户端
│   ├── ollama.py     # check_if_running / query_model / query_generate（本地 Ollama）
│   ├── openai.py     # create_client / run_chatgpt（OPENAI_API_KEY，OPENAI_BASE_URL 可选）
│   └── openrouter.py # create_client / run_openrouter（OPENROUTER_API_KEY）
├── verifiers/        # 答案解析与数学等价验证
│   ├── math.py       # render_prompt / get_last_boxed / extract_final_candidate / grade_answer
│   └── parser.py     # parse_and_verify_answer：advanced 语义解析（区间/集合/矩阵/进制等）
└── judges/           # LLM-as-a-Judge 与排行榜
    ├── llm_judges.py # evaluate_response：统一 judge 入口，query_fn 注入客户端
    ├── ollama.py     # evaluate_response_ollama
    ├── qwen3.py      # evaluate_response_qwen3
    └── leaderboards.py  # Elo（elo_ratings）与 Bradley-Terry（fit_bradley_terry）排行
```

## 与生成侧 / 消费方的分工

| 层 | 职责 | 不负责 |
|---|---|---|
| `evaluation/` | 纯基础设施：调外部服务、解析与评分、judge、排行统计 | 不跑生成循环、不加载本地模型 |
| `inference/harness/` | **生成侧评测循环**：编码→本地模型生成→调用 verifier→统计落盘（[evaluation](../30-inference/evaluation.md)） | 不直接实现解析/评分算法 |
| `post_training/` | **消费方**：`reasoning/data_gen.py`（教师数据 verifier 过滤）、`rl/rewards.py`（RLVR 奖励）、`sft/evaluation/judge.py`（alpaca 指令打分） | 不复制客户端/验证器实现 |

规则：verifier、judge、客户端算法只允许出现在 `evaluation/`；各域只做编排。

## 示例 import

```python
# 客户端与验证器（包级懒加载导出）
from evaluation import query_model, render_prompt, extract_final_candidate, grade_answer

# advanced 解析器（可选依赖 antlr4；缺失自动降级）
from evaluation.verifiers.parser import parse_and_verify_answer

# judge：统一入口，注入任一客户端
from evaluation.judges.llm_judges import evaluate_response
from evaluation.clients.openai import create_client, run_chatgpt
score = evaluate_response(question, response, query_fn=run_chatgpt, reference_answer=gt)

# 排行榜
from evaluation.judges.leaderboards import elo_ratings, fit_bradley_terry
```

## 测试与依赖

- 离线可测：verifier/parser 纯函数、judge 评分协议（注入 query_fn 替身）、Elo/BT 数值——
  `tests/unit/test_llm_judges.py`、`test_leaderboards.py`、`test_math500_harness.py`。
- 外部依赖：sympy（verifiers）；`openai` 客户端与 Ollama 服务、`antlr4-python3-runtime`
  （advanced parser 的 LaTeX 解析）均为运行时可选，缺失时降级或跳过。

## 相关

- [评测 harness（生成侧循环）](../30-inference/evaluation.md)
- [reasoning 训练](../20-post-training/reasoning.md)（RLVR 奖励与蒸馏消费方）
- [测试](../50-development/testing.md)
