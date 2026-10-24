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

2026-10-24 新增（对照 LLMs-from-scratch 补缺）：

- `test_rope_scaling`：模块化 `RoPE` 与函数式实现逐元素等价、buffer 不入 state_dict；
- `test_nf4`：double quantization（scale 二次量化往返、存储 < 单次量化）；
- `test_weight_tying`：GPT-2/Llama3 `tie_word_embeddings` 共享对象同一性、参数量差值、forward；
- `test_kv_sharing`：`SharedKVCache` 组映射、Qwen3 CLA 接线（共享投影 `is` 判断、
  `reuse_kv_only`、cache 两段前向、span=1 行为不变）；
- `test_mtp`：MTPHead 形状/loss/共享参数只计一次；
- `test_tokenizer`：自研 Unigram 训练-编码-解码-存取往返；
- `test_decoding_strategies`：投机解码 draft==target 时与 greedy 逐 token 一致、eos 早停；
- `test_gemma3_loading`：HF 命名合成权重映射、tying 分支、registry 可用性；
- `test_phi15_loading`：Wqkv (3d,d) 按行拆分（含 bias）、partial RoPE 维度、
  tying 兜底、registry 条目；
- `test_rm`：BT 损失方向性、末位非 pad 取分、价值头参数量、
  单 batch 过拟合两步收敛、GPT-2 `output_hidden_states` opt-in 契约；
- `test_nsa`：三分支因果前缀不变性、块可见性/池化/块选择语义、
  压缩门控后与手工滑窗注意力逐元素一致、t<w 分支输出为零；
- `test_decoding_strategies`（beam 追加 4 项）：num_beams=1 与 greedy 逐 token
  一致、多 beam 输出合法、长度惩罚 0/1 选择差异、EOS 冻结早停；
- `test_gated_deltanet`：recurrent 与 chunked 精确并行逐元素等价（含非整除
  chunk/随机初始状态/块边界状态，atol 1e-4）、状态衰减、门控范围、梯度有限；
- `test_mtp`（联合损失追加 6 项）：各深度损失一致性、权重退化、梯度到达主干
  与 MTP 头、短序列报错、hidden opt-in、训练步；
- `test_phi_sft_backbone`：指令 SFT 底座路由（phi15 分支与默认 GPT-2 分支）；
- `test_gemma4`：官方 E2B/E4B layer_types 逐位复现、比例参数化、
  proportional RoPE 三性质、逐层 head_dim、softcap 上界、tying、backward；
- `test_aya`：CAT 三性质（位置 0 恒等/保范数/相对位置内积）+ 仿射生效、
  uvqk 零初始化等价标准 GQA + 四组偏置逐一生效、因果性；
- `test_olmo3` / `test_qwen3_5`：官方配置锁定、参数量解析校验、滑窗窗外
  不变性、delta net 因果性与分块一致、cache 增量=整段、CausalLM 协议。

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
