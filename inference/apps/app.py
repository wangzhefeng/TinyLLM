# -*- coding: utf-8 -*-

# ***************************************************
# * File        : app.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-20
# * Version     : 1.0.102000
# * Description : Chainlit 聊天 UI。
#               用环境变量 APP_MODEL=gpt2|qwen3（默认 gpt2）选择模型：
#                 - gpt2：GPT-2 124M，权重来源用 GPT2_WEIGHTS=orig|own（默认 own）
#                   - orig：官方 GPT-2 124M 权重（下载到 downloaded_models/gpt2）
#                   - own ：自训 checkpoint（环境变量 GPT2_CHECKPOINT，默认 results/checkpoints/gpt2/model.pth）
#                   - 教学模式（设置开关 show_trace）：生成后用
#                     inference/interpretability.trace_generation 展示每步
#                     top-k 候选概率与熵（author="Trace" 的 markdown 表）
#                 - qwen3：经 inference.loading.load_model 加载（默认 key qwen3-0.6b，
#                   QWEN3_MODEL_KEY 可覆盖），format_qwen3_chat 组 prompt，
#                   generate_text_stream 流式输出，<think>...</think> 与回答分离渲染
#               （qwen3 模式 UI 提供 thinking 开关；启动脚本 scripts/run_app.sh）
# * Link        : link
# * Requirement : pip install chainlit（可选依赖，不在主 requirements 中）；
#               gpt2 orig 模式另需 tensorflow（仅读官方 TF checkpoint 时需要）；
#               qwen3 模式另需下载好的 Qwen3 权重与 tokenizer.json
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
# 仓根路径由本文件位置推导（inference/apps/app.py -> 上两级），
# 不依赖启动时的 cwd，保证从任意目录运行都能正确 import 与定位权重
ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

import torch

# chainlit 为可选依赖：未安装时模块仍可导入（便于静态检查/复用），仅 UI handler 不注册
try:
    import chainlit
except ImportError:
    chainlit = None
    print("[app.py] 未安装 chainlit，无法启动聊天 UI。请先 pip install chainlit")

from layers.tokenizers.tokenization import (
    choose_tokenizer,
    text_to_token_ids,
    token_ids_to_text,
)
from models.gpt2.gpt2_124M import Model
from inference.generate_core import generate
from utils.args_tools import DotDict
from utils.device import device_setting

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger

# 模型选择：gpt2（默认）| qwen3
APP_MODEL = os.environ.get("APP_MODEL", "gpt2").lower()
if APP_MODEL not in ("gpt2", "qwen3"):
    raise ValueError(f"APP_MODEL must be 'gpt2' or 'qwen3', got: {APP_MODEL}")

# 权重来源（仅 gpt2 生效）：orig = 官方 GPT-2 124M 权重；own = 自训 checkpoint
WEIGHTS_SOURCE = os.environ.get("GPT2_WEIGHTS", "own").lower()
if WEIGHTS_SOURCE not in ("orig", "own"):
    raise ValueError(f"GPT2_WEIGHTS must be 'orig' or 'own', got: {WEIGHTS_SOURCE}")

# device
device = device_setting(verbose=True)


# ------------------------------
# <think>...</think> 流式分离
# ------------------------------
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


class _ThinkSplitter:
    """把流式生成的文本 fragment 按 <think>/</think> 标签分成两条通道。

    feed(text) 返回 (thinking_text, answer_text)：本次应分别追加到
    thinking 与正式回答的片段；跨 fragment 切开的半个标签会先扣留，
    待下一 fragment 拼齐后再判定（</think> 之后全部归入回答通道）。
    """

    def __init__(self):
        self.in_think = False
        self.closed = False
        self._pending = ""

    def feed(self, text):
        self._pending += text
        out_think, out_answer = "", ""
        while self._pending:
            if self.closed:
                out_answer += self._pending
                self._pending = ""
                break
            marker = THINK_CLOSE if self.in_think else THINK_OPEN
            idx = self._pending.find(marker)
            if idx >= 0:
                chunk = self._pending[:idx]
                if self.in_think:
                    out_think += chunk
                else:
                    out_answer += chunk
                self._pending = self._pending[idx + len(marker):]
                if self.in_think:
                    self.in_think = False
                    self.closed = True
                else:
                    self.in_think = True
                continue
            # 无完整标签：扣留可能是标签前缀的尾部
            keep = 0
            for k in range(min(len(marker) - 1, len(self._pending)), 0, -1):
                if marker.startswith(self._pending[-k:]):
                    keep = k
                    break
            emit = self._pending[:len(self._pending) - keep]
            if emit:
                if self.in_think:
                    out_think += emit
                else:
                    out_answer += emit
                self._pending = self._pending[len(emit):]
            break
        return out_think, out_answer


def _load_gpt2_for_app():
    # model config (gpt2-small 124M)
    model_config = {
        "vocab_size": 50257,     # Vocabular size
        "context_length": 256,  # Context length
        "max_new_tokens": 50,   # Maximum new tokens to generate
        "embed_dim": 768,        # Embedding dimension
        "d_ff": 4 * 768,         # Hidden dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of transformer layers
        "dropout": 0.1,          # Dropout rate
        "qkv_bias": False,       # Query-Key-Value bias
        "dtype": torch.float32,
        "kv_window_size": 1024,  # KV cache window size
        "top_k": 1,
        "temperature": 0.0,
        "eos_id": 50256,
        "use_cache": True,  # GPT-2 已实现 KV cache（models/gpt2/gpt2_124M.py），开启可加速逐 token 生成
        "tokenizer_model": "tiktoken_gpt2_bpe",
    }
    model_config = DotDict(model_config)

    # tokenizer
    tokenizer = choose_tokenizer(tokenizer_model = "tiktoken_gpt2_bpe")

    # model
    if WEIGHTS_SOURCE == "orig":
        # 官方 GPT-2 124M 权重（懒加载：tensorflow 仅在此模式下需要）
        from models.io.gpt2.openai_gpt2_weights_load import (
            download_and_load_gpt2,
            load_weights_download,
        )
        settings, params = download_and_load_gpt2(
            model_size="124M", models_dir=str(Path(ROOT) / "downloaded_models" / "gpt2")
        )
        model = Model(model_config)
        load_weights_download(model, params)
    else:
        # 自训 checkpoint（路径可用环境变量 GPT2_CHECKPOINT 覆盖，默认 results/checkpoints/gpt2/model.pth）
        model_path = Path(os.environ.get("GPT2_CHECKPOINT", str(Path(ROOT) / "results/checkpoints/gpt2/model.pth")))
        if not model_path.exists():
            logger.info(f"Could not find the {model_path} file.")
            sys.exit()
        checkpoint = torch.load(model_path, weights_only=True)
        model = Model(model_config)
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return tokenizer, model, model_config, {}


def _load_qwen3_for_app():
    """Qwen3：经 inference.loading.load_model 加载（registry key 可用环境变量覆盖）。"""
    from inference.loading import load_model
    from models.registry import get_model_spec

    model_key = os.environ.get("QWEN3_MODEL_KEY", "qwen3-0.6b")
    get_model_spec(model_key)
    model, config, extras = load_model(model_key, device=str(device))
    tokenizer = choose_tokenizer("qwen3", tokenizer_file_path=extras["tokenizer_file_path"])
    gen_config = DotDict({
        "tokenizer_model": "qwen3",
        "context_length": config.context_length,
        "max_new_tokens": 2048,
        "temperature": 0.6,
        "top_k": None,
        "top_p": 0.95,
        "eos_id": tokenizer.eos_token_id,
        "use_cache": True,
    })
    return tokenizer, model, gen_config, extras


def get_model_and_tokenizer():
    if APP_MODEL == "qwen3":
        return _load_qwen3_for_app()
    return _load_gpt2_for_app()


if chainlit is not None:
    # Obtain the necessary tokenizer and model files for the chainlit function below
    tokenizer, model, model_config, load_extras = get_model_and_tokenizer()
    logger.info(f"[app.py] APP_MODEL={APP_MODEL} loaded")


    @chainlit.on_chat_start
    async def on_chat_start():
        """会话初始化：qwen3 提供 thinking 开关，gpt2 提供教学模式开关。"""
        chainlit.user_session.set("add_thinking", False)
        chainlit.user_session.set("show_trace", False)
        widgets = []
        if APP_MODEL == "qwen3":
            widgets.append(chainlit.input_widget.Switch(
                "add_thinking", "启用 thinking（生成 <think>...</think> 后再回答）",
                initial=False,
            ))
        else:
            widgets.append(chainlit.input_widget.Switch(
                "show_trace", "教学模式：生成后展示每步 top-k 候选概率与熵",
                initial=False,
            ))
        settings = await chainlit.ChatSettings(widgets).send()
        for key in ("add_thinking", "show_trace"):
            if key in settings:
                chainlit.user_session.set(key, settings[key])


    @chainlit.on_settings_update
    async def on_settings_update(settings):
        for key, value in settings.items():
            chainlit.user_session.set(key, value)


    async def _run_qwen3_stream(content, add_thinking):
        """qwen3：chat 模板组 prompt -> generate_text_stream 流式输出。

        <think>...</think> 段写入 author="Thinking" 的独立消息，
        正式回答写入默认助手消息，两者随生成逐 token 更新。
        """
        from inference.generation import generate_text_stream
        from layers.tokenizers.chat import format_qwen3_chat

        prompt = format_qwen3_chat(
            content, add_generation_prompt=True, add_thinking=add_thinking,
        )
        stream = generate_text_stream(
            model,
            prompt,
            family="qwen3",
            tokenizer_file_path=load_extras.get("tokenizer_file_path"),
            device=str(device),
            max_new_tokens=model_config.max_new_tokens,
            context_length=model_config.context_length,
            temperature=model_config.temperature,
            top_k=model_config.top_k,
            top_p=model_config.top_p,
            eos_token_id=model_config.eos_id,
            use_cache=model_config.use_cache,
        )

        splitter = _ThinkSplitter()
        answer_msg = chainlit.Message(content="")
        await answer_msg.send()
        think_msg = None
        try:
            for fragment in stream:
                think_text, answer_text = splitter.feed(fragment)
                if think_text:
                    if think_msg is None:
                        think_msg = chainlit.Message(content="", author="Thinking")
                        await think_msg.send()
                    await think_msg.stream_token(think_text)
                if answer_text:
                    await answer_msg.stream_token(answer_text)
        finally:
            if think_msg is not None:
                await think_msg.update()
            await answer_msg.update()


    # 测试代码 main 函数
    @chainlit.on_message
    async def main(message: chainlit.Message):
        """
        The Main Chainlit function
        """
        if APP_MODEL == "qwen3":
            await _run_qwen3_stream(
                message.content,
                add_thinking=chainlit.user_session.get("add_thinking") or False,
            )
            return
        token_ids = text_to_token_ids(
            message.content,
            tokenizer_model=model_config.tokenizer_model
        ).to(device)
        if chainlit.user_session.get("show_trace"):
            # 教学模式：trace_generation 逐步记录 top-k/熵（无 cache 教学路径）
            from inference.interpretability import render_trace_md, trace_generation

            records, out_ids = trace_generation(
                model,
                token_ids,
                max_new_tokens=model_config.max_new_tokens,
                context_length=model_config.context_length,
                temperature=model_config.temperature,
                top_k=model_config.top_k,
                eos_token_id=model_config.eos_id,
                k=5,
                tokenizer=tokenizer,
            )
            text = token_ids_to_text(
                out_ids, tokenizer_model=model_config.tokenizer_model
            )
            await chainlit.Message(content=f"{text}").send()
            await chainlit.Message(
                content=render_trace_md(records), author="Trace",
            ).send()
            return
        token_ids = generate(
            model=model,
            token_ids=token_ids,
            max_new_tokens=model_config.max_new_tokens,
            context_length=model_config.context_length,
            temperature=model_config.temperature,
            top_k=model_config.top_k,
            eos_token_id=model_config.eos_id,
            use_cache=model_config.use_cache,
        )
        text = token_ids_to_text(token_ids, tokenizer_model=model_config.tokenizer_model)

        # Returns the model response to the interface
        await chainlit.Message(content=f"{text}").send()
else:
    def main(*args, **kwargs):
        raise RuntimeError(
            "chainlit 未安装，无法启动聊天 UI。请先 pip install chainlit，"
            "再用 inference/scripts/run_app.sh [orig|own] [gpt2|qwen3] 启动。"
        )

if __name__ == "__main__":
    main()
