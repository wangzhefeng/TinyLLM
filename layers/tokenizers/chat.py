def format_input_alpaca(entry):
    instruction = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry.get("input") else ""
    return instruction + input_text


def format_input_phi3(entry):
    instruction = f"<|user|>\n{entry['instruction']}"
    return instruction[:-1] + f": '{entry['input']}'"


def format_qwen3_chat(user_text, add_generation_prompt=False, add_thinking=False):
    text = f"<|im_start|>user\n{user_text}<|im_end|>\n"
    if add_generation_prompt:
        text += "<|im_start|>assistant"
        text += "\n" if add_thinking else "\n<think>\n\n</think>\n\n"
    return text


def format_gemma3_chat(user_text):
    return f"<start_of_turn>user\n{user_text}<end_of_turn>\n<start_of_turn>model\n"


def apply_chat_template(family, user_text, **kwargs):
    templates = {
        "qwen3": format_qwen3_chat,
        "gemma3": format_gemma3_chat,
    }
    key = family.lower()
    if key not in templates:
        available = ", ".join(sorted(templates))
        raise ValueError(f"Unknown chat family '{family}'. Available: {available}")
    return templates[key](user_text, **kwargs)
