# layers/ 与 models/ 模块审查（2026-10-20）

> 所属：[TinyLLM Docs](../../README.md) › [参考与分析](../index.md)

对 `layers/` 与 `models/` 两个模块的完整代码审查 + 对照上游两本书的差距分析。
结论口径：以"代码是否可运行、数值是否正确"为准，全部问题均经源码逐条验证。

- **对照对象**：[rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)（`main`，含 42 个 bonus）、[rasbt/reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch)（`main`，8 章 + 附录）
- **审查范围**：`layers/` 全部 27 个 Python 文件（约 5900 行）、`models/` 全部 13 个文件，并交叉核对 `config/`、`post_training/`、`inference/`
- **分析日期**：2026-10-20
- **历史对照**：[llms-from-scratch 旧版差距分析](../llms-from-scratch/index.md)（2026-10-18 快照，路径口径已失真，本报告为其刷新版）

## 一页结论

1. **代码健康度不足以支撑继续学习**：发现 11 组运行时必挂或输出必错的 P0 bug（含 attention 缩放指数符号写反、Qwen3-optimized cache 路径条件写反、Gemma3 未做权重 tying 等），另有大量"改了一个副本漏了另一个"的重复实现 drift。
2. **预训练视角**：核心层（attention/GQA/RoPE/RMSNorm/MoE）基本齐全，但注意力变体只有 GQA 落地——MLA 是空壳、SWA 锁死在 Gemma3 block 内、无跨层 KV 共享 / DeltaNet / DSA；DeepSeek 模型整体是 `NotImplementedError` 骨架。
3. **推断视角**：采样（greedy/temperature/top-k/top-p）、KV cache、batch + early stop、streaming 已具备；缺 beam search、repetition penalty、推断时扩展（self-consistency / self-refinement / Best-of-N 全是占位文件）、KV cache 与 SWA/MLA 的联动。
4. **后训练视角**：reasoning-from-scratch 主线（ch04–ch08）基本未实现——GRPO from scratch、蒸馏均为 42 行占位；post_training/preference 下的 GRPO 依赖 unsloth/vllm，不是 from-scratch 教学实现。
5. **修复顺序**：先修 P0 + 补回归测试（否则一切学习都建立在错误代码上）→ 结构整改（改名/归位/去 sys.path hack/统一接口）→ 按三域补教学主题。

## 分页

| 页面 | 内容 |
|---|---|
| [gaps.md](gaps.md) | 按预训练/推断/后训练列出的缺失主题清单 |
| [bugs.md](bugs.md) | 已验证的代码问题：P0（必修）/ P1（结果错误）/ P2（设计缺陷） |
| [roadmap.md](roadmap.md) | 三阶段落地路线与验收标准 |

## 相关

- [LLMs-from-scratch 旧分析（2026-10-18 快照）](../llms-from-scratch/index.md)
- [仓库架构](../../00-overview/architecture.md)
- [代码→文档映射](../../00-overview/repo-map.md)
