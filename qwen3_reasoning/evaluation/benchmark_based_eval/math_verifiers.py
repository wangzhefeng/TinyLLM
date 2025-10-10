# -*- coding: utf-8 -*-

# ***************************************************
# * File        : math_verifiers.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-10-05
# * Version     : 1.0.100516
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import re
import json
from urllib.request import urlopen
import warnings
warnings.filterwarnings("ignore")

from sympy.parsing import sympy_parser as spp
from sympy.core.sympify import SympifyError
from sympy import simplify
from datasets import load_dataset

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


# ------------------------------
# Extract final answer box
# ------------------------------
def get_last_boxed(text):
    # Find the last occurrence of "\boxed"
    boxed_start_idx = text.rfind(r"\boxed")
    if boxed_start_idx == -1:
        return None

    # Get position after "\boxed"
    current_idx = boxed_start_idx + len(r"\boxed")

    # Skip any whitespace after "\boxed"
    while current_idx < len(text) and text[current_idx].isspace():
        current_idx += 1

    # Expect an opening brace "{"
    if current_idx >= len(text) or text[current_idx] != "{":
        return None

    # Parse the braces with nesting
    current_idx += 1
    brace_depth = 1
    content_start_idx = current_idx

    while current_idx < len(text) and brace_depth > 0:
        char = text[current_idx]
        if char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
        current_idx += 1

    # Account for unbalanced braces
    if brace_depth != 0:
        return None

    # Extract content inside the outermost braces
    return text[content_start_idx:current_idx-1]


def extract_final_candidate(text, fallback="number_then_full"):
    """
    extract final candidate

    Args:
        text (_type_): _description_
        fallback (str, optional): 如果没有找到框式内容，则使用备用设置. Defaults to "number_then_full".
            - "number_then_full": 选择最后一个简单数字，否则选择整个文本
            - "number_only": 选择最后一个简单数字，否则返回一个空字符串 ""
            - "none": 仅提取框内内容，否则返回空字符串 ""

    Returns:
        _type_: _description_
    """
    # Default return value if nothing matches
    result = ""
    if text:
        # Prefer the last boxed expression if present
        boxed = get_last_boxed(text.strip())
        
        if boxed:
            result = boxed.strip().strip("$ ")
        # If no boxed expression, try fallback
        elif fallback in ("number_then_full", "number_only"):
            RE_NUMBER = re.compile(r"-?(?:\d+/\d+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
            m = RE_NUMBER.findall(text)
            if m:
                # Use last number
                result = m[-1]
            elif fallback == "number_then_full":
                # Else return full text if no number found
                result = text
    
    return result

# ------------------------------
# Normalizing the extracted answer
# ------------------------------
def normalize_text(text):
    LATEX_FIXES = [  # Latex formatting to be replaced
        (r"\\left\s*", ""),
        (r"\\right\s*", ""),
        (r"\\,|\\!|\\;|\\:", ""),
        (r"\\cdot", "*"),
        (r"\u00B7|\u00D7", "*"),
        (r"\\\^\\circ", ""),
        (r"\\dfrac", r"\\frac"),
        (r"\\tfrac", r"\\frac"),
        (r"°", ""),
    ]

    RE_SPECIAL = re.compile(r"<\|[^>]+?\|>")  # strip chat special tokens like <|assistant|>

    if not text:
        return ""
    text = RE_SPECIAL.sub("", text).strip()

    # Remove angle-degree markers
    text = re.sub(r"\^\s*\{\s*\\circ\s*\}", "", text)   # ^{\circ}
    text = re.sub(r"\^\s*\\circ", "", text)             # ^\circ
    text = text.replace("°", "")                        # Unicode degree

    # unwrap \text{...} if the whole string is wrapped
    match = re.match(r"^\\text\{(?P<x>.+?)\}$", text)
    if match:
        text = match.group("x")

    # strip inline/display math wrappers \( \) \[ \]
    text = re.sub(r"\\\(|\\\)|\\\[|\\\]", "", text)

    # light LaTeX canonicalization
    for pat, rep in LATEX_FIXES:
        text = re.sub(pat, rep, text)

    # numbers/roots
    text = text.replace("\\%", "%").replace("$", "").replace("%", "")
    text = re.sub(
        r"\\sqrt\s*\{([^}]*)\}",
        lambda match: f"sqrt({match.group(1)})",
        text,
    )
    text = re.sub(
        r"\\sqrt\s+([^\\\s{}]+)",
        lambda match: f"sqrt({match.group(1)})",
        text,
    )

    # fractions
    text = re.sub(
        r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
        lambda match: f"({match.group(1)})/({match.group(2)})",
        text,
    )
    text = re.sub(
        r"\\frac\s+([^\s{}]+)\s+([^\s{}]+)",
        lambda match: f"({match.group(1)})/({match.group(2)})",
        text,
    )

    # exponent and mixed numbers
    text = text.replace("^", "**")
    text = re.sub(
        r"(?<=\d)\s+(\d+/\d+)",
        lambda match: "+" + match.group(1),
        text,
    )

    # 1,234 -> 1234
    text = re.sub(
        r"(?<=\d),(?=\d\d\d(\D|$))",
        "",
        text,
    )

    return text.replace("{", "").replace("}", "").strip().lower()

# ------------------------------
# Verifying mathematical equivalence
# ------------------------------
def sympy_parser(expr):
    try:
        return spp.parse_expr(
            expr,
            transformations=(
                # Standard transformations like handling parentheses
                *spp.standard_transformations,

                # Allow omitted multiplication symbols (e.g., "2x" -> 2*x")
                spp.implicit_multiplication_application,
            ),
            # Evaluate during parsing so simple constants simplify (e.g., 2+3 -> 5)
            evaluate=True,
        )
    except (SympifyError, SyntaxError, TypeError, IndexError):
        return None


def equality_check(expr_gtruth, expr_pred):
    # First, check if the two expressions are exactly the same string
    if expr_gtruth == expr_pred:
        return True

    # Parse both expressions into SymPy objects (returns None if parsing fails)
    gtruth, pred = sympy_parser(expr_gtruth), sympy_parser(expr_pred)

    # If both expressions were parsed successfully, try symbolic comparison
    if gtruth is not None and pred is not None:
        try:
            # If the difference is 0, they are equivalent
            return simplify(gtruth - pred) == 0
        except (SympifyError, TypeError):
            pass

    return False

# ------------------------------
# Grading answers
# ------------------------------
def split_into_parts(text):
    result = [text]

    if text:
        # Check if text looks like a tuple or list, e.g. "(a, b)" or "[a, b]"
        if (
            len(text) >= 2
            and text[0] in "([" and text[-1] in ")]"
            and "," in text[1:-1]
        ):
            # Split on commas inside brackets and strip whitespace
            items = [p.strip() for p in text[1:-1].split(",")]
            if all(items):
                result = items
    else:
        # If text is empty, return an empty list
        result = []

    return result


def grade_answer(pred_text, gt_text):
    result = False  # Default outcome if checks fail

    # Only continue if both inputs are non-empty strings
    if pred_text is not None and gt_text is not None:
        gt_parts = split_into_parts(
            normalize_text(gt_text)
        )  # Break ground truth into comparable parts

        pred_parts = split_into_parts(
            normalize_text(pred_text)
        )  # Break prediction into comparable parts

        # Ensure both sides have same number of valid parts
        if (gt_parts and pred_parts
           and len(gt_parts) == len(pred_parts)):
            result = all(
                equality_check(gt, pred)
                for gt, pred in zip(gt_parts, pred_parts)
            )  # Check each part for mathematical equivalence

    return result  # True only if all checks passed


def run_demos_table(tests):
    header = ("Test", "Expect", "Got", "Status")
    rows = []
    for name, pred, gtruth, expect in tests:
        got = grade_answer(pred, gtruth)  # Run equality check
        status = "PASS" if got == expect else "FAIL"
        rows.append((name, str(expect), str(got), status))

    data = [header] + rows
    
    # Compute max width for each column to align table nicely
    col_widths = [
        max(len(row[i]) for row in data)
        for i in range(len(header))
    ]

    # Print table row by row
    for row in data:
        line = " | ".join(
            row[i].ljust(col_widths[i])
            for i in range(len(header))
        )
        print(line)

    # Print summary of passed tests
    passed = sum(r[3] == "PASS" for r in rows)
    print(f"\nPassed {passed}/{len(rows)}")

# ------------------------------
# Load dataset
# ------------------------------
# data download
hf_data = load_dataset("HuggingFaceH4/MATH-500", split="test", cache_dir="./dataset")

# data path
math_data_path = Path("./dataset/math500_test.json")

# data write
with open(math_data_path, "w", encoding="utf-8") as f:
    json.dump(hf_data.to_list(), f, ensure_ascii=False, indent=2)

# data load
if math_data_path.exists():
    with math_data_path.open("r", encoding="utf-8") as f:
        math_data = json.load(f)
else:
    url = (
        "https://raw.githubusercontent.com/rasbt/reasoning-from-scratch/"
        "main/ch03/01_main-chapter-code/math500_test.json"
    )
    with urlopen(url) as f:
        math_data = json.load(f)
logger.info(f"Number of entries: {len(math_data)}")
from pprint import pprint
pprint(math_data[0])

# ------------------------------
# Evaluating Model
# ------------------------------
def render_prompt(prompt):
    template = (
        "You are a helpful math assistant.\n"
        "Answer the question and write the final result on a new line as:\n"
        "\\boxed{ANSWER}\n\n"
        f"Question:\n{prompt}\n\nAnswer:"
    )
    return template






# ------------------------------
# test
# ------------------------------
'''
# model answer
model_answer = (
r"""....some explanation...
**Final Answer:**

\[
\boxed{\dfrac{14}{3}}
\]
""")
extracted_answer = get_last_boxed(model_answer)
logger.info(f"extracted_answer: {extracted_answer}")

extracted_answer = extract_final_candidate(model_answer)
logger.info(f"extracted_answer: {extracted_answer}")

extracted_answer = extract_final_candidate(text=r"\boxed{14/3. }")
logger.info(f"extracted_answer: {extracted_answer}")

extracted_answer = extract_final_candidate(text="abc < > 14/3 abc")
logger.info(f"extracted_answer: {extracted_answer}")

extracted_answer = extract_final_candidate(text="Text without numbers")
logger.info(f"extracted_answer: {extracted_answer}")

logger.info(normalize_text(extract_final_candidate(model_answer)))
logger.info(normalize_text(r"$\dfrac{14}{3.}$"))
logger.info(normalize_text(r"\text{\[\frac{14}{3}\]}"))
logger.info(normalize_text("4/3"))


logger.info(sympy_parser(normalize_text(extract_final_candidate(model_answer))))
logger.info(sympy_parser("28/6"))
logger.info(equality_check(normalize_text("13/4."), normalize_text(r"(13)/(4)")))
logger.info(equality_check(normalize_text("0.5"), normalize_text(r"(1)/(2)")))
logger.info(equality_check(normalize_text("14/3"), normalize_text("15/3")))
logger.info(equality_check(normalize_text("(14/3, 2/3)"), normalize_text("(14/3, 4/6)")))

split_into_parts(normalize_text(r"(14/3, 2/3)"))
grade_answer("14/3", r"\frac{14}{3}")
grade_answer(r"(14/3, 2/3)", "(14/3, 4/6)")

# Define test cases: (name, prediction, ground truth, expected result)
tests = [
        ("check_1", "3/4", r"\frac{3}{4}", True),
        ("check_2", "(3)/(4)", r"3/4", True),
        ("check_3", r"\frac{\sqrt{8}}{2}", "sqrt(2)", True),
        ("check_4", r"\( \frac{1}{2} + \frac{1}{6} \)", "2/3", True),
        ("check_5", "(1, 2)", r"(1,2)", True),
        ("check_6", "(2, 1)", "(1, 2)", False),
        ("check_7", "(1, 2, 3)", "(1, 2)", False),
        ("check_8", "0.5", "1/2", True),
        ("check_9", "0.3333333333", "1/3", False),
        ("check_10", "1,234/2", "617", True),
        ("check_11", r"\text{2/3}", "2/3", True),
        ("check_12", "50%", "1/2", False),
        ("check_13", r"2\cdot 3/4", "3/2", True),
        ("check_14", r"90^\circ", "90", True),
        ("check_15", r"\left(\frac{3}{4}\right)", "3/4", True),
    ]
run_demos_table(tests)
'''



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
