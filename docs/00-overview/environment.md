> 所属：[TinyLLM Docs](../README.md) › [Overview](index.md)

# Environment Setup

本项目使用 [uv](https://docs.astral.sh/uv/) 管理 Python 虚拟环境（Python 3.12，见 `.python-version`）。

```bash
# 创建/同步虚拟环境（读取 pyproject.toml 与 uv.lock）
uv sync

# 激活虚拟环境
source .venv/bin/activate

# 或者直接用 uv run 运行脚本
uv run python run_gpt2_124M.py
```

- `pyproject.toml` / `uv.lock` 中固定的是 `torch==2.5.1` 的 CPU/MPS 版本，适用于 macOS（Apple Silicon 上自动使用 MPS）。
- Linux + NVIDIA 环境如需 CUDA 12.1 版本，请改用 `requirements.txt`：

```bash
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

- 若 `~/.cache/uv` 不可写（例如受限沙箱环境），可在全局配置 `~/.config/uv/uv.toml` 中设置 `cache-dir = "/tmp/uv-cache"`，对所有项目一次性生效，无需每次 `export`。
