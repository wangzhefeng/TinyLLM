> 所属：[TinyLLM Docs](../../README.md) › [Post-training](../index.md) › [Fine-tuning](index.md)

## 避免过拟合与欠拟合

### 过拟合

> 过于专业

模型记住了训练数据，无法泛化到未见过的输入。解决方案：

* 如果训练时间短，降低学习率(learning rate)。对于较长的训练，提高学习率(learning ratge)。
  因此，最好测试两者，看看哪个效果更好。
* 增加批处理大小(batch size)。
* 降低训练轮数(epochs)。
* 将你的数据集与通用数据集合并，例如 ShareGPT
* 增加 dropout 率以引入正则化。

### 欠拟合

> 过于泛化

尽管不常见，欠拟合是指低秩模型由于缺乏可学习参数而无法泛化，因此您的模型可能无法从训练数据中学习。解决方案：

* 如果训练时间短，请提高学习率(learning rate)。对于较长的训练运行，请降低学习率(learning rate)。
* 增加 `rank` 和 `alpha`。`alpha` 应至少等于 `rank` 数，而 `rank` 对于较小的模型/更复杂的数据集应更大；
  它通常在 `4` 到 `64` 之间。
* 使用更符合领域相关性的数据集。

## 训练和评估

### 训练

一旦所有设置完成，就可以开始训练了！如果遇到问题，记得你可以随时调整超参数、数据集等。

训练过程中你会看到一些数字的日志！这是训练损失，你的任务是调整参数使它尽可能接近 0.5！
如果你的微调没有达到 1、0.8 或 0.5，你可能需要调整一些数值。如果损失变为 0，那可能也不是一个好迹象！

我们通常建议保持默认设置，除非你需要更长时间的训练或更大的批量大小。

* `per_device_train_batch_size = 2`：增加以提升 GPU 利用率，但需注意因填充导致的训练速度变慢。
  相反，增加 `gradient_accumulation_steps` 以实现更平滑的训练。
* `gradient_accumulation_steps = 4`：模拟更大的批处理大小，而不会增加内存使用。
* `max_steps = 60`：加速训练。对于完整运行，替换为 `num_train_epochs = 1` （建议 1-3 个 epoch 以避免过拟合）。
* `learning_rate = 2e-4`：降低以实现更慢但更精确的微调。尝试 `1e-4`、`5e-5` 或 `2e-5` 等值。

### 评估

为了评估，你可以通过直接与模型聊天来进行手动评估，看看是否符合你的喜好。
你也可以启用 Unsloth 的评估功能，但要注意，这可能会根据数据集的大小而耗时。
为了加快评估速度，你可以：减少评估数据集的大小或设置 `evaluation_steps = 100`。

对于测试，你也可以取你训练数据的 20% 来用于测试。如果你已经用完了所有训练数据，那么你就需要手动评估它。
你也可以使用自动评估工具，如 EleutherAI 的 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)。
要注意的是，自动化工具可能无法完全符合你的评估标准。

## 模型推理和保存

### 模型推理

现在让我们在完成训练过程后运行模型！你可以编辑黄色下划线部分！实际上，
因为我们创建了一个多轮聊天机器人，我们现在也可以像它过去看到一些对话那样调用模型，如下所示：

提醒 Unsloth 本身也提供了原生 2 倍的推理速度，所以永远不要忘记调用 `FastLanguageModel.for_inference(model)`。
如果你想让模型输出更长的响应，将 `max_new_tokens = 128` 设置为 256 或 1024 等更大的数字。
请注意，你也需要等待更长时间才能得到结果！

### 模型保存

为了在 Ollama、vLLM、Open WebUI 等期望的推理引擎中保存和使用您的模型，我们这里可以提供更多信息：

* https://docs.unsloth.ai/basics/running-and-saving-models

现在我们可以将微调后的模型保存为一个名为 LoRA 适配器的 100MB 小文件，如下所示。
如果您想上传您的模型，也可以将其推送到 Hugging Face Hub！
记得通过 https://huggingface.co/settings/tokens 获取 Hugging Face 令牌并添加您的令牌！

保存模型后，我们再次可以使用 Unsloth 来运行模型！使用 `FastLanguageModel` 再次调用它进行推理！
