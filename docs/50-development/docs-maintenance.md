> 所属：[TinyLLM Docs](../README.md) › [Development](index.md)

# 文档维护规范

## 1. 基本原则

1. **唯一文档中心**：项目级文档全部放 `docs/`；代码目录只放代码。
2. **渐进式披露**：入口页只给概要和目录，细节拆到子页面。
3. **单页单主题**：每页尽量 ≤ 150 行；超过就拆页。
4. **链接优先**：不复制内容，用相对链接指向唯一来源。
5. **根 README 是总索引**：只放项目定位、三域入口、快速开始和文档入口。
6. **术语统一**：`inference` 用“模型推断/推断”，`reasoning` 用“逻辑推理/推理”；详见 [术语约定](../00-overview/terminology.md)。

## 2. 目录约定

```text
docs/
├── README.md              # 文档总索引
├── 00-overview/           # 项目概览、架构、环境、代码→文档映射
├── 10-pretraining/        # 预训练
├── 20-post-training/      # 后训练 / reasoning
├── 30-inference/          # 模型推断
├── 40-models/             # 模型与 tokenizer
├── 50-development/        # 测试、数据、清理、维护
├── 60-reference/          # 书本分析、迁移、学习资料
└── archive/               # 二进制/历史材料（非正文）
```

## 3. 页面结构

每个正文页面建议包含：

```markdown
# 主题

> 所属：[TinyLLM Docs](../README.md) › [域](index.md)

## 概要
...

## 细节
...

## 相关
- [相关页面](...)
```

## 4. 修改代码时的文档动作

| 代码变更 | 必须更新 |
|---|---|
| 新增/删除/移动功能 | 对应域页面、域 `index.md`、[repo-map](../00-overview/repo-map.md) |
| 改命令/参数/依赖 | 对应操作页、根 README 的快速开始（如有） |
| 改目录结构 | 根 README 目录树、[architecture](../00-overview/architecture.md)、repo-map |
| 改测试 | [testing](testing.md) |
| 改数据路径 | [datasets](datasets.md) |
| 删除内容 | [cleanup-log](cleanup-log.md) |

## 5. 巡检（人也需要定期检查）

规则不能保证 100% 执行，建议：

- **每次 PR/提交前**：检查新增/修改代码是否同步文档；检查链接。
- **每月**：扫描超过 150 行的文档、孤立页面、失效链接。
- **每季度**：对照 [repo-map](../00-overview/repo-map.md) 做一次“代码→文档”覆盖审计。

可用命令：

```bash
# 找出超过 150 行的文档
find docs -name '*.md' -exec wc -l {} + | awk '$1>150 && $2!="total"'

# 检查相对链接（示例：用 grep 找 markdown 链接后人工核对）
grep -Rno '](\.\./[^)]*)' docs | head
```

## 6. 归档

- 历史分析、迁移记录放 `docs/60-reference/`。
- 二进制/历史 bundle 放 `docs/archive/`。
- 归档内容也要有入口页链接，不能成为“孤儿文件”。

## 7. 相关

- [AGENTS.md](../../AGENTS.md)
- [代码→文档映射](../00-overview/repo-map.md)
- [文档总索引](../README.md)
