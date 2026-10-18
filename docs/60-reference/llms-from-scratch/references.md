> 所属：[TinyLLM Docs](../../README.md) › [LLMs-from-scratch 分析](index.md)

## 9. 附：关键路径索引

### 仓库侧

- 总览：`README.md`
- 训练主线：`ch05/01_main-chapter-code/ch05.ipynb`、`gpt_train.py`
- 分类：`ch06/01_main-chapter-code/gpt_class_finetune.py`
- 指令微调：`ch07/01_main-chapter-code/gpt_instruction_finetuning.py`
- DPO：`ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb`
- LoRA：`appendix-E/01_main-chapter-code/appendix-E.ipynb`
- 训练循环增强：`appendix-D/01_main-chapter-code/appendix-D.ipynb`
- 注意力变体：`ch04/03_kv-cache` … `ch04/10_kv-sharing`
- 模型 zoo：`ch05/11_qwen3` … `ch05/17_gemma4`
- 可安装包：`pkg/llms_from_scratch/`
- 推理书：`reasoning-from-scratch/`

### 项目侧

- 预训练闭环：`run_gpt2_124M.py`、`exp/exp_pretrain_gpt2_124M.py`
- 层与注意力：`layers/attention.py`、`layers/kv_cache.py`、`layers/moe.py`
- 模型：`models/gpt2/`、`models/llama/`、`models/qwen3/`、`models/gemma3/`
- 数据：`data_provider/data_loader.py`、`data_provider/pgcorpus_gutenberg/`
- 训练工具：`utils/optimizer.py`、`utils/learning_rate.py`、`utils/calc_loss.py`
- 分布式工具库：`tools/distributed/`
- 微调（已归档）：`todo/finetuning/`
- UI / 推理 demo（已归档）：`todo/inference/`
- 推理模型占位：`qwen3_reasoning/`
- 迁移总览：`docs/00-overview/architecture.md`、`docs/README.md`

---

## 10. 参考链接

- 仓库：<https://github.com/rasbt/LLMs-from-scratch>
- 书页：<https://www.manning.com/books/build-a-large-language-model-from-scratch>
- 推理书：<https://github.com/rasbt/reasoning-from-scratch>
- PyPI 包：<https://pypi.org/project/llms-from-scratch/>
- 作者博客（架构对比）：<https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison>
- 作者博客（KV cache）：<https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms>
