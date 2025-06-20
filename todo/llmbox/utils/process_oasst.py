import json
import sys

from model_training.custom_datasets.oasst_dataset import load_oasst_export


def from_tree_to_conversation(input_path, save_path):
    train, eval = load_oasst_export(input_file_path=input_path)
    with open(save_path, "w", encoding="utf-8") as f:
        for entry in train:
            text_list = entry.get_formatted(eos_token="")
            text_list = [text.replace("<|prompter|>", "").replace("<|assistant|>", "") for text in text_list]
            res_dict = {"conversations": text_list}
            json.dump(res_dict, f)
            f.write("\n")
