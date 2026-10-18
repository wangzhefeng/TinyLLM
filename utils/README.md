# utils

TinyLLM 项目内部工具包。原先是与作者其他仓库共用的独立 `utils` 仓库（通过
`PYTHONPATH` 或直接 checkout 到项目根目录的方式引用），其中包含大量与本项目无关的
CV / 时序 / 可视化工具。这里按 TinyLLM 的实际 import 做了最小化重建。

## 目录

```
utils/
├── args_tools.py       # DotDict、print_args_llm
├── calc_loss.py        # calc_loss_batch / calc_loss_loader
├── criterion.py        # select_criterion
├── ddp_utils.py        # 分布式初始化 / rank / DDP
├── device.py           # device_setting、torch_gc
├── early_stopping.py   # EarlyStopping
├── learning_rate.py    # adjust_learning_rate / warmup
├── load_save_data.py   # load_local_data / load_hf_data
├── log_util.py         # logger（控制台 + 按天轮转文件）
├── model_memory.py     # 参数量 / 显存估算
├── optimizer.py        # select_optimizer
├── plot_losses.py      # loss 曲线（matplotlib 可选）
├── random_seed.py      # set_seed / set_seed_torch
└── timestamp_utils.py  # unix 时间戳转换
```

> 原先 `utils/llm/` 子包已扁平化到 `utils/` 下，import 路径由
> `utils.llm.xxx` 改为 `utils.xxx`。

## 来源

大部分文件直接来自共享仓库 `~/projects/utils`；`args_tools.py`、`timestamp_utils.py`
在共享仓库中已缺失，取自同一作者的 `~/projects/llm_proj/nlpproj/utils`。

## 与上游的差异（为适配 TinyLLM / macOS / 当前依赖）

1. `log_util.py`：`LOG_NAME` 未设置时使用 `default`，不再生成 `logs/None/`。
2. `args_tools.py`：`print_args_llm` 改为遍历参数对象输出，避免上游硬编码
   `emb_dim` / `iters` / `use_multi_gpu` 等字段与 `run_*.py` 不一致时抛 AttributeError。
3. `optimizer.py`：`galore_torch` 改为按需导入；AdamW 的 `fused` 仅在 CUDA 下开启。
4. `plot_losses.py`：matplotlib 改为函数内延迟导入，未安装时告警并跳过绘图。
5. `model_memory.py`：兼容 `tok_embed` / `tok_emb` 两种命名，CUDA 显存统计仅在可用时读取。
6. `ddp_utils.py`：后端按 CUDA 可用性选择（nccl/gloo），`torch.cuda.set_device`
   加保护；新增 `ddp_setup` / `ddp_setup_custom`；修复 `USE_LIBUV` 的 KeyError。
7. `random_seed.py`：同时提供 `set_seed` 与 `set_seed_torch`，CUDA 调用加保护。
8. `early_stopping.py`：DDP 下仅 rank 0 写 checkpoint。
9. `learning_rate.py`：未知 `lradj` 时不再触发 `UnboundLocalError`。
10. 目录扁平化：上游 `utils/llm/*` 移到 `utils/*`，import 路径同步改为 `utils.xxx`。

## 未纳入的模块

共享仓库中的 `cv/`、`visual/`、`pysci/`、`torch_operation/`、`online/`、`templates`
等与 TinyLLM 无关，未复制。若后续代码需要（例如 HF/W&B 登录），可按需补充。
