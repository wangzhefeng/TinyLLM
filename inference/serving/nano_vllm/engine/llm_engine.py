# -*- coding: utf-8 -*-

# ***************************************************
# * File        : llm_engine.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-28
# * Version     : 1.0.072816
# * Description : LLMEngine: nano_vllm 对外入口。
# *               ├── add_request(): prompt -> Sequence 进入 waiting 队列
# *               ├── step(): 一次调度 + 前向 + 后处理(continuous batching 核心循环)
# *               ├── generate(): 循环 step 直到所有请求完成
# *               └── 多进程 TP: tensor_parallel_size > 1 时 spawn tp-1 个
# *                   worker 子进程(ModelRunner rank>0)，经 SharedMemory 收发命令
# *               tokenizer 可选(传入即可输出文本，否则只输出 token ids)。
# * Link        : https://github.com/GeeeekExplorer/nano-vllm
# * Requirement : torch >= 2.0
# ***************************************************

__all__ = [
    "LLMEngine",
]

# python libraries
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import atexit
import os
import socket
from time import perf_counter

import torch.multiprocessing as mp

from ..config import Config
from ..sampling_params import SamplingParams
from .sequence import Sequence
from .scheduler import Scheduler
from .model_runner import ModelRunner

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def _find_free_port() -> int:
    """动态选择一个空闲 tcp 端口(供 gloo init 使用，避免硬编码端口冲突)"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LLMEngine:

    def __init__(self, config: Config, model=None, tokenizer=None, verbose: bool = False):
        assert isinstance(config, Config)
        self.config = config
        Sequence.block_size = config.kvcache_block_size
        self.tokenizer = tokenizer
        self.verbose = verbose
        self.ps: list[mp.Process] = []
        self.events: list = []
        tp = config.tensor_parallel_size
        if tp > 1 and model is not None:
            # 外部模型只存在于主进程；tp>1 时 worker 子进程会各自随机初始化，
            # 与 rank0 的外部模型权重不一致，必须显式拒绝而非静默出错
            raise ValueError(
                "tensor_parallel_size > 1 时不支持传入外部 model："
                "worker 子进程无法共享主进程的模型权重。"
                "请用 Config.from_pretrained + 默认随机初始化，或改用 tensor_parallel_size=1。"
            )
        shm_name = f"nano_vllm_{os.getpid()}" if tp > 1 else None
        if tp > 1:
            if config.tp_master_port == -1:
                config.tp_master_port = _find_free_port()
            # spawn 上下文启动 tp-1 个 worker 子进程(子进程继承父进程 sys.path，
            # spawn 目标 ModelRunner 位于包内模块，可正常 import)
            ctx = mp.get_context("spawn")
            for i in range(1, tp):
                event = ctx.Event()
                process = ctx.Process(target=ModelRunner, args=(config, i, event, shm_name))
                process.start()
                self.ps.append(process)
                self.events.append(event)
        # rank0 在主进程；tp=1 时无 shm/event，纯单进程
        self.model_runner = ModelRunner(config, 0, self.events, shm_name, model)
        self.scheduler = Scheduler(config)
        if tp > 1:
            atexit.register(self.exit)

    def exit(self):
        """
        退出：通知所有 worker 退出命令循环 -> 释放 shm -> join 子进程，不留僵尸进程。
        幂等：显式调用与 atexit 重复触发时只执行一次。
        """
        if getattr(self, "_exited", False):
            return
        self._exited = True
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            assert self.tokenizer is not None, "传入文本 prompt 需要提供 tokenizer"
            prompt = self.tokenizer.encode(prompt)
        assert len(prompt) >= 1 and len(prompt) < self.config.max_model_len
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self) -> tuple[list[tuple[int, list[int]]], int]:
        """
        执行一个 step：
        返回 (本 step 完成的请求 [(seq_id, completion_token_ids)], 本 step 处理的 token 数)
        token 数为正表示 prefill，为负表示 decode(绝对值为序列数)
        """
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        if self.verbose:
            phase = "prefill" if is_prefill else "decode "
            detail = ", ".join(f"seq{seq.seq_id}(+{seq.num_scheduled_tokens}tok)" for seq in seqs)
            print(f"  [{phase}] batch={len(seqs)}: {detail}")
        # tp>1 时经 shm 通知所有 worker 同步执行同一 step
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self) -> bool:
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        verbose: bool | None = None,
    ) -> list[dict]:
        """
        批量生成，返回 [{"text": ..., "token_ids": ...}, ...](按请求顺序)
        """
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        old_verbose, self.verbose = self.verbose, (self.verbose if verbose is None else verbose)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_tokens = decode_steps = 0
        t0 = perf_counter()
        while not self.is_finished():
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_tokens += num_tokens
            else:
                decode_steps += 1
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
        elapsed = perf_counter() - t0
        self.verbose = old_verbose
        results = []
        for seq_id in sorted(outputs.keys()):
            token_ids = outputs[seq_id]
            text = self.tokenizer.decode(token_ids) if self.tokenizer is not None else None
            results.append({"text": text, "token_ids": token_ids})
        if old_verbose:
            print(f"  完成 {len(results)} 条请求: prefill {prefill_tokens} tokens, "
                  f"decode {decode_steps} steps, 耗时 {elapsed:.2f}s")
        return results




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
