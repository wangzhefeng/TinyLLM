# AGENTS.md

本文件是 TinyLLM 的**强制协作规则**，适用于人类与 AI Agent。  
目标：代码可维护，文档不腐化。

## 1. 文档系统（强制）

1. **唯一文档中心**：项目级文档全部放在 `docs/`。  
   - 允许的例外：根目录 `README.md`（总索引）与本文件。
   - 代码目录（`pre_training/`、`post_training/`、`inference/`、`models/`、`layers/` 等）**不再放长 README**。
2. **根 README 只做索引**：项目定位、三域入口、快速开始、文档入口。不要塞长篇笔记。
3. **渐进式披露**：
   - 入口页（`docs/README.md`、各域 `index.md`）只放概要与目录；
   - 每个正文页尽量 **≤ 150 行**，超过就拆页；
   - 页面之间用相对链接连接，不复制内容。
4. **单页单主题**：一个页面只解释一个主题；相关背景用链接。
5. **导航头**：正文页顶部写
   ```markdown
   > 所属：[TinyLLM Docs](../README.md) › [域](index.md)
   ```
6. **命名**：小写、短横线；入口统一 `index.md`；分析/历史放 `docs/60-reference/`；二进制放 `docs/archive/` 或 `docs/assets/`。

## 2. 修改功能时必须更新文档

任何代码改动都必须检查对应文档，不能只改代码：

| 变更 | 必须更新 |
|---|---|
| 新增/删除/移动功能 | 对应域页面、域 `index.md`、[repo-map](docs/00-overview/repo-map.md) |
| 改命令/参数/依赖 | 对应操作页、根 README 快速开始（如受影响） |
| 改目录结构 | 根 README 目录树、[architecture](docs/00-overview/architecture.md)、repo-map |
| 改测试 | [testing](docs/50-development/testing.md) |
| 改数据路径 | [datasets](docs/50-development/datasets.md) |
| 删除内容 | [cleanup-log](docs/50-development/cleanup-log.md) |
| 新增外部参考 | [reference-projects](docs/50-development/reference-projects.md) |

**完成标准**：代码可运行 + 相关文档已更新 + 链接可跳转 + 必要时更新索引。

## 3. 提交/评审清单

- [ ] 是否新增/移动/删除了代码目录？→ 更新 repo-map 与 architecture。
- [ ] 是否改了命令或参数？→ 更新对应域页面。
- [ ] 是否新增了文档？→ 加到域 `index.md` 与 `docs/README.md`。
- [ ] 是否有超过 150 行的文档？→ 拆分。
- [ ] 是否有代码目录下的 README/长 markdown？→ 移到 `docs/`。
- [ ] 相对链接是否有效？→ 运行 `scripts/check_docs.py`（见下）。
- [ ] 是否产生重复说明？→ 合并到唯一页面并用链接引用。

## 4. 定期巡检（人类也要做）

规则可能遗漏，建议：

- **每次提交前**：跑测试 + 文档检查脚本。
- **每月**：检查超大文档、孤立页面、失效链接。
- **每季度**：按 [repo-map](docs/00-overview/repo-map.md) 做一次“代码→文档”覆盖审计。

命令：

```bash
python -m unittest discover -s tests -v
python scripts/check_docs.py
```

`check_docs.py` 会报告：

- 超过 150 行的文档；
- 指向不存在文件的相对链接；
- `docs/` 外的 markdown 文档（根 README/AGENTS 除外）。

## 5. 文档事实来源

- 目录/职责：`docs/00-overview/architecture.md`
- 代码映射：`docs/00-overview/repo-map.md`
- 三域入口：`docs/10-pretraining/`、`docs/20-post-training/`、`docs/30-inference/`
- 模型：`docs/40-models/`
- 测试/数据/清理：`docs/50-development/`
- 上游分析/迁移：`docs/60-reference/`
- 独立工具库：`docs/70-tools/`

若文档与代码冲突，以**代码 + 测试**为准，并立即修正文档；不要保留“过期但看起来正确”的描述。

## 6. 禁止事项

- 禁止在 `pre_training/`、`post_training/`、`inference/`、`models/` 等代码目录新增长 README；
- 禁止复制整段文档到多个页面；
- 禁止只改代码不改文档；
- 禁止删除文档页面而不更新索引/链接；
- 禁止把数据、权重、日志提交进 git（见 `.gitignore`）。
