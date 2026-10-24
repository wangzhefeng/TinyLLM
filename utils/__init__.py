"""
TinyLLM 内部工具包（vendored from the author's shared `utils` repository）。

本包只保留 TinyLLM 实际 import 到的模块，便于项目独立运行：

- 通用：`log_util`、`args_tools`、`device`、`random_seed`、`timestamp_utils`
- 训练：`model_memory`、`plot_losses`、`calc_loss`、`criterion`、`optimizer`、
  `early_stopping`、`learning_rate`、`load_save_data`
- 分布式：`ddp_utils`

原先的 `utils/llm/` 子包已扁平化到 `utils/` 下，import 路径由 `utils.llm.xxx`
改为 `utils.xxx`。

训练专用新代码使用顶层 `training/`；本包保留通用工具和旧 API 兼容。
"""

__all__ = []
