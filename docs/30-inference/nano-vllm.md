> 所属：[TinyLLM Docs](../README.md) › [模型推断](index.md)

# nano_vllm：迷你 vLLM 推断引擎

教学级 vLLM 复刻，移植自 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)，
改造为**纯 PyTorch、CPU 可跑**（triton / flash-attn 替换为 SDPA，CUDA 专用路径做守卫式保留）。
支持**多进程张量并行（TP）**：gloo 后端 + 共享内存命令通道，CPU 上即可跑 tp=2。

## 结构

```text
inference/serving/nano_vllm/
├── config.py            # Config：模型结构 + 引擎调度参数
├── sampling_params.py   # temperature / top_k / max_tokens / ignore_eos
├── layers/              # TP 线性层、SDPA attention、RoPE、RMSNorm、MLP、sampler
├── models/              # Qwen3ForCausalLM（compute_logits、tie embeddings）
├── engine/
│   ├── sequence.py      # 请求状态机 WAITING/RUNNING/FINISHED + block_table
│   ├── block_manager.py # 分页 KV block：free/used、引用计数、前缀缓存 hash
│   ├── scheduler.py     # continuous batching：chunked prefill + 抢占
│   ├── model_runner.py  # KV cache 分配、prefill/decode batch 构造、TP worker 循环、CUDA graph
│   └── llm_engine.py    # LLMEngine：add_request / step / generate，tp>1 时 spawn worker
├── utils/               # Context（调度元信息）、权重加载/随机初始化
├── example.py           # 随机权重微型 Qwen3 + 伪 tokenizer 的闭环演示
├── example_pretrained.py# 加载真实 HF 格式 Qwen3 权重推断
└── bench.py             # 吞吐基准（随机小模型或真实权重）
```

## 运行

```python
# 作为库使用（包化，无副作用，不遮蔽仓根同名包）
from inference.serving.nano_vllm import LLM, Config, SamplingParams
```

```bash
# 随机权重演示（无需下载）：chunked prefill + continuous batching
.venv/bin/python inference/serving/nano_vllm/example.py   # 或 python -m inference.serving.nano_vllm.example

# 吞吐基准（默认随机微型模型；--model-path 指向 HF 目录则加载真实权重）
.venv/bin/python inference/serving/nano_vllm/bench.py --num-seqs 32
.venv/bin/python inference/serving/nano_vllm/bench.py --model-path ~/huggingface/Qwen3-0.6B/

# 真实权重推断（HF 格式目录，可用 models/io/qwen3/model_download.py 下载）
.venv/bin/python inference/serving/nano_vllm/example_pretrained.py \
    --model-path ~/huggingface/Qwen3-0.6B/ "你好"
```

- 随机权重演示无需下载：展示 chunked prefill + continuous batching。
- 数值正确性：paged KV 输出与无 cache 朴素重算 allclose（误差 ~2e-7）。
- 真实权重路径（`Config.from_pretrained` 读 config.json + `load_model` 读 safetensors，
  含 q/k/v、gate/up 打包映射）有单元测试：`tests/unit/test_nano_vllm.py`。

## 张量并行（TP）

`Config(tensor_parallel_size=N)`（1–8）：tp>1 时 LLMEngine 以 spawn 启动 N-1 个 worker 子进程，
rank0 经共享内存（`SharedMemory` + Event，1MB 命令缓冲）广播 `run/exit`，各 rank 用 gloo 通信；
权重按参数名确定性播种生成全局权重，再经 `weight_loader` 分片，保证各 rank 一致。
CUDA graph 仅 `cuda 可用且 enforce_eager=False` 时启用（GPU 路径未在本仓 CI 验证，CPU 恒 eager）。

```python
config = Config(vocab_size=64, hidden_size=128, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                tensor_parallel_size=2)   # heads/kv_heads/vocab/intermediate 需被 tp 整除
```

正确性测试 `tests/unit/test_nano_vllm_tp.py`：tp=1 与 tp=2 贪心生成逐 token 一致。
限制：tp>2 未实测；多进程下不支持传入外部 model 实例；shm 命令缓冲固定 1MB。

## 与上游的取舍

- triton store_kvcache kernel → `index_copy_`；flash-attn → `F.scaled_dot_product_attention`
- 多进程 TP 的 nccl/CUDA → gloo/CPU；CUDA graph 改为守卫式（GPU 可用时才启用）
- transformers AutoConfig/AutoTokenizer → 内嵌 Config + 可选 tokenizer 注入
- 采样器在上游 temperature 基础上补了逐序列 top_k 与贪心分支

## 相关

- [serving](serving.md)：推断原理与部署形态
- [loaders](loaders.md)：真实 Qwen3 权重加载（教材版模型）
