> 所属：[TinyLLM Docs](../README.md) › [Development](index.md)

# 测试

项目使用标准库 `unittest`，不额外引入 pytest。测试使用 tiny 配置，快速层不需要网络或真实大模型权重。

## 运行

```bash
# 全量
python -m unittest discover -s tests -v

# 分层运行
python -m unittest discover -s tests/unit -v
python -m unittest discover -s tests/integration -v

# 额外启用多进程 gloo 冒烟
TINYLLM_SLOW_TESTS=1 python -m unittest discover -s tests -v
```

CI 位于 `.github/workflows/tests.yml`：pull request 运行 unit、integration 和 docs check；push 额外运行慢速分布式测试。CUDA/NCCL 和真实外部服务不属于默认离线测试。

## 目录与覆盖

| 目录 | 内容 |
|---|---|
| `unit/` | layers、models、tokenizer、RL、PEFT、tools、推断、配置与兼容 shim |
| `integration/` | tiny 预训练→checkpoint→resume、指令 SFT、DPO、Gutenberg 可选上游测试 |

重点契约测试包括：

- GPT-2/Qwen3 cache 与无 cache 数值等价、两个 cache 交错隔离；
- 模型/生成/tokenizer/config 公共协议与新旧模块路径同源；
- nano_vllm RoPE、TP 一致性、可变 top-k、外部模型 TP 限制；
- `training.Trainer` 梯度累积、AMP 设备行为、评估模式恢复；
- checkpoint 统一 schema 与历史 `optmizer` 兼容；
- SFT/DPO 默认数据路径、DPO 输出命名、统一 CLI；
- MoE eval 确定性、ALiBi dtype、weight tying 参数统计。

## 集成测试约定

- `test_tiny_pretrain_resume.py` 使用合成文本真实执行两轮训练与 latest 恢复。
- `test_tiny_post_training.py` 用本地 tiny 模型替代外部权重下载，真实执行 SFT/DPO 前向、反向、评估和保存。
- `test_pgcorpus_gutenberg.py` 依赖可选上游 `llms_from_scratch`，未安装时跳过。
- CUDA-only 路径在无 CUDA 环境跳过；不能用 CPU/gloo 结果宣称 NCCL 已验证。

## 修改测试时

新增公开入口必须有 CLI/help 或 smoke 覆盖；修改模型 forward/cache、checkpoint schema、目录路径时必须添加契约测试。不得通过删除测试或降低数值断言来迁就重构。

文档链接检查：

```bash
python scripts/check_docs.py
```
