# TinyLLM 文档索引

> 本目录是 TinyLLM 的**唯一文档中心**。根 [README](../README.md) 只做项目总索引；这里按主题逐步展开。  
> 维护规则见 [docs-maintenance](50-development/docs-maintenance.md) 与根目录 [AGENTS.md](../AGENTS.md)。

## 按阶段

| 阶段 | 入口 | 说明 |
|---|---|---|
| 预训练 | [10-pretraining/index.md](10-pretraining/index.md) | 数据、训练、示例、Gutenberg、extras |
| 后训练 / reasoning | [20-post-training/index.md](20-post-training/index.md) | SFT、PEFT、DPO、RLHF、蒸馏、reasoning |
| 模型推断 | [30-inference/index.md](30-inference/index.md) | loaders、nano_vllm、评测、UI、reasoning scaling |

## 横切主题

| 主题 | 入口 |
|---|---|
| 项目概览 / 架构 | [00-overview/index.md](00-overview/index.md) |
| 模型 | [40-models/index.md](40-models/index.md) |
| 开发 / 测试 / 数据 / 清理 | [50-development/index.md](50-development/index.md) |
| 参考 / 分析 / 迁移 | [60-reference/index.md](60-reference/index.md) |
| 工具库 | [70-tools/index.md](70-tools/index.md) |
| 归档 | [archive/](archive/)（二进制/历史材料） |

## 渐进式披露约定

1. 每个页面只讲一个主题，尽量控制在 **150 行以内**。
2. `index.md` / 域入口只放概览、目录和链接，不重复细节。
3. 细节页顶部放“所属”导航，底部放“相关”链接。
4. 代码目录不放长文档；新增内容统一写到本目录。
5. 功能变更必须同步更新对应文档；详见 [docs-maintenance](50-development/docs-maintenance.md)。

## 常用入口

- [术语约定：推断 vs 推理](00-overview/terminology.md)
- [环境搭建](00-overview/environment.md)
- [仓库架构](00-overview/architecture.md)
- [代码→文档映射](00-overview/repo-map.md)
- [测试](50-development/testing.md)
- [数据集](50-development/datasets.md)
- [清理记录](50-development/cleanup-log.md)
- [layers/models 审查（2026-10-20）](60-reference/layers-models-audit/index.md)
- [LLMs-from-scratch 分析](60-reference/llms-from-scratch/index.md)
- [迁移记录](60-reference/migration-2026-10.md)
