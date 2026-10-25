> 所属：[TinyLLM Docs](../README.md) › [模型推断](index.md)

# 评测

评测设施分两层：**算法与外部服务上提**到公共库
[evaluation/](../35-evaluation/index.md)（客户端/验证器/judge/排行榜）；
**生成侧评测循环**保留在本域 `inference/harness/`。本页只讲 harness。

## harness（生成侧评测循环）

| 文件 | 内容 |
|---|---|
| `inference/harness/math500.py` | MATH-500 端到端：生成 → 解析 → 评分 → 统计落盘 |
| `inference/harness/mmlu.py` | MMLU 选择题三种 predict 方法与 `evaluate_mmlu` |

数据下载和模型加载在调用时发生，核心纯函数可离线单测；MATH/MMLU 与外部 judge 需要对应缓存、网络或服务。

## MATH-500 端到端 harness

`math500.py` 把 `render_prompt → 编码 → 生成 → extract_final_candidate → grade_answer → 统计` 串成评测循环，报告 JSON 落盘（`save_report` 自动建目录）。`generate_fn` / `batched_generate_fn` 可注入替身，评测循环完全离线可单测；默认批量路径内部走 `generate_qwen3_batched_cache`（右填充 + attn_mask 屏蔽 pad）。

```python
from inference.harness import evaluate_math500, evaluate_math500_batched

report = evaluate_math500(model, tokenizer, problems, generate_fn=my_gen_fn, cot=True)
report = evaluate_math500_batched(model, tokenizer, problems, batch_size=4)
report["accuracy"], report["num_problems"], report["results"][0]
```

CLI（策略：greedy/cot/self-consistency/weighted-sc/best-of-n/self-refinement，重依赖懒加载）：

```bash
.venv/bin/python -m inference.harness.math500 \
    --model qwen3-0.6b --strategy cot --subset 50 --parser advanced --out results/math500/cot.json
```

- `--n` 为采样条数（self-consistency/weighted-sc/best-of-n）或最大修订轮数（self-refinement）；
- `--temperature` 只作用于采样类策略，greedy/cot 固定贪心；
- `--parser basic|advanced`（默认 basic）：评分器选择，见下节；
- weighted-sc 默认用书 ch05 的 `heuristic_score` 加权投票，分数全 0 时退化为多数票；
- 批量跑五个策略并汇总 accuracy：`bash inference/scripts/run_math500_experiments.sh`（`MODEL`/`SUBSET`/`N`/`OUTDIR` 等环境变量可覆盖）。

## 答案评分器：basic vs advanced

评分算法位于公共库 `evaluation/verifiers/`（见 [评测公共库](../35-evaluation/index.md)）：

- **basic**（默认）：`evaluation/verifiers/math.py` 的 `extract_final_candidate`
  取最后一个 `\boxed` + `grade_answer`（`normalize_text` 规范化 + sympy 数值等价）。
- **advanced**：`evaluation/verifiers/parser.py` 的 `parse_and_verify_answer`，移植自
  [rasbt/reasoning-from-scratch bonus parser](https://github.com/rasbt/reasoning-from-scratch/blob/main/reasoning_from_scratch/bonus/parser.py)。
  在 basic 之外额外支持区间、集合、矩阵、元组/逗号列表、`\pm` 多答案、进制下标、
  货币等结构的语义解析。`parse_latex` 依赖可选包 `antlr4-python3-runtime`：未安装或
  解析失败时自动降级（sympify 路径 / basic `grade_answer` 兜底），离线环境仍可用。

## MMLU 多项选择：三种 predict 方法

`inference/harness/mmlu.py`（书 chF/02）：

1. `predict_choice`：生成后逐字符找字母（Method 1，实测最弱）；
2. `predict_choice_logprob`：first-token logprob 比较（Method 2，推荐）；
3. `predict_choice_teacher_forcing`：teacher-forcing（Method 3）——对每个选项的
   continuation `X. <choice>` 逐 token 计算平均 log-prob（NLL 最小者胜出）；
   `choices=None` 时退化为单字母 token 打分面。

```python
from inference.harness.mmlu import evaluate_mmlu, predict_choice_teacher_forcing

results = evaluate_mmlu(
    model, tokenizer, data,
    predict_fn=lambda m, t, p: predict_choice_teacher_forcing(m, t, p)[0],
)
```

## 相关

- [评测公共库](../35-evaluation/index.md)
- [serving](serving.md)
- [reasoning](../20-post-training/reasoning.md)
- [数据集](../50-development/datasets.md)
