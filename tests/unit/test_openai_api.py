# -*- coding: utf-8 -*-

# ***************************************************
# * File        : test_openai_api.py
# * Description : inference/api/openai_api.py 凭证读取回归测试：
# *               从环境变量 OPENAI_API_KEY / OPENAI_BASE_URL 读取，
# *               未设置 key 时抛出带指引的 RuntimeError（不再读 config.json）。
# * Requirement : 无（openai 包未安装时仅跑缺 key 用例）
# ***************************************************

import os
import unittest
from unittest import mock

from inference.api import openai_api


class TestCreateClient(unittest.TestCase):

    def test_missing_api_key_raises_clear_error(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                openai_api.create_client()

    @unittest.skipIf(openai_api.OpenAI is None, "openai 包未安装")
    def test_client_uses_env_credentials(self):
        env = {
            "OPENAI_API_KEY": "sk-test-123",
            "OPENAI_BASE_URL": "http://localhost:9999/v1",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            client = openai_api.create_client()
        self.assertEqual(client.api_key, "sk-test-123")
        self.assertEqual(str(client.base_url).rstrip("/"), "http://localhost:9999/v1")

    @unittest.skipIf(openai_api.OpenAI is None, "openai 包未安装")
    def test_client_without_base_url(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-123"}, clear=True):
            client = openai_api.create_client()
        self.assertEqual(client.api_key, "sk-test-123")


if __name__ == "__main__":
    unittest.main()
