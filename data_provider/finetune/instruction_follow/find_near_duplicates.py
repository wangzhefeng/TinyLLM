# -*- coding: utf-8 -*-

# ***************************************************
# * File        : find_near_duplicates.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-18
# * Version     : 0.1.021823
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import re
import argparse

from data_provider.load_save_data import load_json_data, save_json_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def _preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    return text


def find_near_duplicates(json_data, key="instruction", threshold=0.75):
    """
    The higher the threshold, the more similar the texts have to be to match
    """
    # Extract instructions
    text = [_preprocess_text(item[key]) for item in json_data if item[key]]
    near_duplicates = []
    indices_to_remove = set()

    if not text:
        return {}, near_duplicates

    # Vectorize the text data
    vectorizer = TfidfVectorizer(stop_words=None, analyzer='char', ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform(text)

    # Compute cosine similarity between each pair of entries
    cos_sim_matrix = cosine_similarity(tfidf_matrix)

    # Find pairs of near-duplicate instructions based on the threshold
    for i in range(len(cos_sim_matrix)):
        for j in range(i+1, len(cos_sim_matrix)):
            if cos_sim_matrix[i, j] > threshold:
                if len(json_data[i][key]) <= 1 or len(json_data[j][key]) <= 1:
                    continue
                near_duplicates.append((json_data[i], json_data[j], cos_sim_matrix[i, j]))
                if key in ("input", "output"):  # Don't remove duplicates based on the instruction
                    indices_to_remove.add(j)  # Mark the second entry for removal
    # Remove the near-duplicate entries
    filtered_json_data = [
        item 
        for index, item in enumerate(json_data) 
        if index not in indices_to_remove
    ]

    return filtered_json_data, near_duplicates


def find_print_and_remove_near_duplicates(json_data, remove_duplicates=False, threshold=0.75):
    """
    Searches each key in the first JSON object for duplicates across a list of JSON objects.
    Prints the duplicates if found.
    """
    for key in json_data[0].keys():
        if remove_duplicates:
            json_data, near_duplicates = find_near_duplicates(json_data, key=key, threshold=threshold)
        else:
            _, near_duplicates = find_near_duplicates(json_data, key=key, threshold=threshold)
       
        separator = 50 * '='
        print(f"\n\n{separator}\nSearching '{key}' for duplicates ...\n{separator}")
        if not near_duplicates:
            print("No duplicates found")
        else:
            for dup in near_duplicates:
                print(
                    f"Duplicate pair found with similarity {dup[2]:.2f}:\n"
                    f"1. {dup[0][key]}\n2. {dup[1][key]}\n"
                )
    
    return json_data

 
def args_parse():
    """
    define and parse command arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", type=str, required=True,
                        default="./dataset/finetune/instruction-example.json",
                        help=("Path to the dataset JSON file"))
    parser.add_argument("--threshold", type=float, 
                        default=0.9,
                        help=("A sensitivity threshold between 0 and 1 where 1 is strictest"))
    parser.add_argument("--remove_duplicates", type=bool, required=True, 
                        default=False,
                        help=("Removes duplicates based on the 'input' or 'output' keys "
                              " (but not the 'instruction') and saves the cleaned JSON file as --json_output_file"))
    parser.add_argument("--json_output_file", type=str, required=True,
                        default="./dataset/finetune/instruction-example-without-duplicates.json",
                        help="Path to the dataset JSON file")
    args = parser.parse_args()

    if args.remove_duplicates and not args.json_output_file:
        raise ValueError(
            "Provide an output file via --json_output_file "
            "to save the cleaned JSON data."
        )
    
    return args




# 测试代码 main 函数
def main():
    # command arguments
    args = args_parse()

    # data load
    json_data = load_json_data(args.json_file)
    # logger.info(f"json_data: \n{json_data}")

    # data preprocess
    json_data = find_print_and_remove_near_duplicates(
        json_data = json_data,
        remove_duplicates = args.remove_duplicates,
        threshold = args.threshold
    )
    # logger.info(f"json_data: \n{json_data}")
    
    # data save
    if args.remove_duplicates:
        save_json_data(json_data, args.json_output_file)
    logger.info(f"JSON data saved to {args.json_output_file}")

if __name__ == "__main__":
    main()
