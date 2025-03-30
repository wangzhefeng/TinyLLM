
# Meta Llama model download

## 模型下载文档

* https://www.llama.com/llama-downloads/
* https://www.llama.com/llama-downloads/

## 模型下载

1. 安装 llama-stack

```bash
$ uv pip install llama-stack
$ uv pip install llama-stack -U
```

2. 查看模型列表

```bash
$ llama model list
$ llama model list --show all
```

3. 下载模型

```bash
$ llama model download --source meta --model-id MODEL_ID
```

当脚本需要输入唯一的字定义 URL 时，粘贴 URL `.env/Llama_model_url`。
