# -*- coding: utf-8 -*-

# ***************************************************
# * File        : model_runner.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-28
# * Version     : 1.1.102020
# * Description : ModelRunner: 模型执行器。
# *               ├── allocate_kv_cache(): 分配分页 KV cache 并挂载到各 Attention 层
# *               ├── prepare_prefill()/prepare_decode(): 把调度器输出的 Sequence
# *               │   列表转换为模型输入(input_ids/positions/slot_mapping/block_tables)，
# *               │   并通过 set_context() 传递给 Attention 层
# *               ├── run(): 一次 step = 准备输入 -> 前向 -> 采样 -> 返回 token ids
# *               ├── 多进程 TP(tensor_parallel_size > 1)：
# *               │   rank0 在主进程，rank>0 由 LLMEngine spawn 为子进程，
# *               │   通过 SharedMemory + Event 接收命令(loop/read_shm)，
# *               │   分布式后端为 gloo(CPU 可跑)
# *               └── CUDA graph(capture_cudagraph)：仅 cuda 可用且非
# *                   enforce_eager 时启用；注意 GPU 路径未在本仓 CI 验证
# * Link        : https://github.com/GeeeekExplorer/nano-vllm
# * Requirement : torch >= 2.0
# ***************************************************

__all__ = [
    "ModelRunner",
]

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
PKG_ROOT = str(Path(__file__).resolve().parents[1])
if PKG_ROOT in sys.path:
    sys.path.remove(PKG_ROOT)
sys.path.insert(0, PKG_ROOT)
import warnings
warnings.filterwarnings("ignore")

import pickle
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event

import torch
import torch.distributed as dist

from config import Config
from engine.sequence import Sequence
from layers.sampler import Sampler
from models.qwen3_for_causal_lm import Qwen3ForCausalLM
from utils.context import set_context, get_context, reset_context
from utils.loader import init_random_weights

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class ModelRunner:

    def __init__(
        self,
        config: Config,
        rank: int = 0,
        event: Event | list[Event] = None,
        shm_name: str | None = None,
        model: Qwen3ForCausalLM | None = None,
    ):
        """
        Args:
            config: 全局配置
            rank: TP rank。0 在主进程，>0 在 LLMEngine spawn 的子进程中
            event: rank0 为所有 worker 的 Event 列表，worker 为自己的单个 Event
            shm_name: 进程间命令通道的 SharedMemory 名(tp>1 时必填)
            model: 可选外部模型；默认按 Config 随机初始化(各 rank 结果一致)
        """
        self.config = config
        self.block_size = config.kvcache_block_size
        # worker 进程内 pickle 反序列化出的 Sequence 依赖该类属性计算 block 边界
        Sequence.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.shm_name = shm_name

        if self.world_size > 1:
            # gloo 后端，CPU 可做真实张量并行
            dist.init_process_group(
                "gloo",
                init_method=f"tcp://127.0.0.1:{config.tp_master_port}",
                world_size=self.world_size,
                rank=rank,
            )
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(config.dtype)
        # 未显式传入模型时按 Config 构造并随机初始化；init_random_weights 按参数名
        # 播种生成全局权重再分片，保证各 rank 从同一全局权重切出本分片
        self.model = model or init_random_weights(Qwen3ForCausalLM(config))
        self.model.eval()
        self.sampler = Sampler()
        self.use_cuda_graph = torch.cuda.is_available() and not self.enforce_eager
        self.warmup_model()
        self.allocate_kv_cache()
        if self.use_cuda_graph:
            # 注意：GPU 路径未在本仓 CI 验证(本机无 CUDA)，仅守卫式移植上游逻辑
            self.capture_cudagraph()
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name=self.shm_name, create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name=self.shm_name)
                self.loop()  # worker 常驻命令循环，直到收到 "exit"

    # ------------------------------
    # 多进程命令通道(移植自上游)
    # ------------------------------
    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
            dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4: n + 4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        assert n + 4 <= self.shm.size, "shm 命令缓冲区溢出"
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4: n + 4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    # ------------------------------
    # 初始化
    # ------------------------------
    def warmup_model(self):
        """
        预热：用假序列跑一次 prefill(此时 KV cache 未分配，store_kvcache 自动跳过)
        """
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len, 64)
        num_seqs = min(max(max_num_batched_tokens // seq_len, 1), self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)

    def allocate_kv_cache(self):
        """
        分配分页 KV cache：
        kv_cache shape = [2(K/V), num_layers, num_blocks, block_size, num_kv_heads/tp, head_dim]
        并把每一层的 K/V 视图挂载到对应 Attention 模块上
        """
        config = self.config
        num_kv_heads = config.num_key_value_heads // self.world_size
        self.kv_cache = torch.zeros(
            2,
            config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            config.head_dim,
        )
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
        assert layer_id == config.num_hidden_layers

    # ------------------------------
    # 输入准备
    # ------------------------------
    def prepare_block_tables(self, seqs: list[Sequence]) -> torch.Tensor:
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        return torch.tensor(block_tables, dtype=torch.int32)

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end  # 该序列可见的上下文长度(含历史缓存前缀)
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            if not seq.block_table:  # warmup 场景
                continue
            # 计算本 step 每个 token 写入 KV cache 的槽位: block_id * block_size + 块内偏移
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        input_ids = torch.tensor(input_ids, dtype=torch.int64)
        positions = torch.tensor(positions, dtype=torch.int64)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32)
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:  # 存在前缀缓存命中
            block_tables = self.prepare_block_tables(seqs)
        set_context(True, cu_seqlens_q, cu_seqlens_k, slot_mapping=slot_mapping, block_tables=block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)  # 当前 token 的位置 id
            context_lens.append(len(seq))   # 可见上下文长度(含当前 token)
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64)
        positions = torch.tensor(positions, dtype=torch.int64)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32)
        context_lens = torch.tensor(context_lens, dtype=torch.int32)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = torch.tensor([seq.temperature for seq in seqs], dtype=torch.float32)
        top_ks = torch.tensor([seq.top_k for seq in seqs], dtype=torch.int64)
        return temperatures, top_ks

    # ------------------------------
    # 前向 + 采样
    # ------------------------------
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        if is_prefill or not self.use_cuda_graph or input_ids.size(0) > 512:
            hidden_states = self.model(input_ids, positions)
            if is_prefill:
                # 只对每条序列的最后一个位置计算 logits
                cu_seqlens_q = get_context().cu_seqlens_q
                last_indices = (cu_seqlens_q[1:] - 1).long()
                hidden_states = hidden_states[last_indices]
            return self.model.compute_logits(hidden_states)
        else:
            # CUDA graph replay 路径(仅 GPU；注意未在本仓 CI 验证)
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int] | None:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        if self.rank == 0:
            temperatures, top_ks = self.prepare_sample(seqs)
        else:
            temperatures = top_ks = None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures, top_ks).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    # ------------------------------
    # CUDA graph(守卫式移植上游，仅 GPU；未在本仓 CI 验证)
    # ------------------------------
    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        max_bs = min(config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
