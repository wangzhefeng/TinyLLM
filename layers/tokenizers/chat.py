def format_input_alpaca(entry):
    instruction = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry.get("input") else ""
    return instruction + input_text


def format_qwen3_chat(user_text, add_generation_prompt=False, add_thinking=False):
    text = f"<|im_start|>user\n{user_text}<|im_end|>\n"
    if add_generation_prompt:
        text += "<|im_start|>assistant"
        text += "\n" if add_thinking else "\n<think>\n\n</think>\n\n"
    return text
