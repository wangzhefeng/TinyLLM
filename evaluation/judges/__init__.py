# -*- coding: utf-8 -*-

# ***************************************************
# * File        : __init__.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-25
# * Version     : 1.0.102511
# * Description : LLM-as-a-Judge 子包：llm_judges（纯打分逻辑）/
# *               ollama（本地 Ollama 后端）/ qwen3（本地 transformers
# *               后端）/ leaderboards（Elo 与 Bradley-Terry 排行榜）。
# *               各模块独立 import，均不在此处预加载（leaderboards
# *               依赖 torch，预加载会拖累整个子包的 import）。
# * Requirement : 无
# ***************************************************
