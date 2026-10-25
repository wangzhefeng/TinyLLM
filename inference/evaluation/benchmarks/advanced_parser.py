# -*- coding: utf-8 -*-

# ***************************************************
# * File        : advanced_parser.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102421
# * Description : MATH-500 混合 LaTeX 答案解析器（advanced parser）。
# *               移植自 reasoning-from-scratch bonus 材料（issue #133，
# *               原始实现 by @labdmitriy）：在基础 normalize_text 之外
# *               额外支持区间、集合、矩阵、元组、\pm、\in、进制下标、
# *               货币、多答案 FiniteSet 等结构的语义解析与等价比较。
# *               parse_latex 依赖 antlr4-python3-runtime（可选依赖）：
# *               未安装或解析失败时自动降级（sympify 路径 / basic 评分），
# *               本模块保持"导入零副作用"。
# * Link        : https://github.com/rasbt/reasoning-from-scratch/blob/main/reasoning_from_scratch/bonus/parser.py
# * Requirement : sympy；antlr4-python3-runtime（可选，启用 parse_latex）
# ***************************************************

__all__ = [
    "sympy_parser_hybrid",
    "normalize_text_hybrid",
    "parse_and_verify_answer",
]

# python libraries
import os
import re
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

from sympy import Contains, FiniteSet, Interval, Matrix, Symbol, Union, oo, sympify
from sympy import simplify, zeros
from sympy.matrices import MatrixBase

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL


# parse_latex 依赖 antlr4-python3-runtime，属可选依赖：
# 懒加载并缓存可用性，缺失/解析失败一律返回 None（不抛出）
_LATEX_PARSER_AVAILABLE = None


def _latex_parse(text):
    global _LATEX_PARSER_AVAILABLE
    if _LATEX_PARSER_AVAILABLE is False:
        return None
    try:
        from sympy.parsing.latex import parse_latex
    except ImportError:
        _LATEX_PARSER_AVAILABLE = False
        return None
    _LATEX_PARSER_AVAILABLE = True
    try:
        parsed = parse_latex(text)
    except Exception:
        return None
    # parse_latex 可能返回未求值表达式（如 2/3 -> Mul(2, Pow(3, -1))），
    # 统一 simplify 成规范形，保证与 sympify 结果可比较
    try:
        return simplify(parsed)
    except Exception:
        return parsed


RE_TEXT = re.compile(r"^\\text\s*\{([^}]*)\}$")
RE_MBOX = re.compile(r"^\\mbox\s*\{([^}]*)\}$")
RE_CURRENCY = re.compile(r"^\\\$\s*([0-9,.\s\\!]+)$")
RE_MATRIX = re.compile(
    r"\\begin\{[pbvBV]?matrix\}(.*)\\end\{[pbvBV]?matrix\}", re.DOTALL
)
RE_TUPLE_LIKE = re.compile(r"^[\(\[\{].*,.*[\)\]\}]$", re.DOTALL)
RE_INTERVAL_BOUNDS = re.compile(r"^([\(\[])(.+),(.*)([\)\]])$")
RE_LEADING_DOT = re.compile(r"^(\.)(\d+)$")
RE_INFTY = re.compile(r"[+-]?\\infty")
RE_MEMBERSHIP = re.compile(r"^(?P<lhs>.+?)\\in(?P<rhs>.+)$", re.DOTALL)
RE_PM = re.compile(r"^(?P<a>.+?)\\pm(?P<b>.+)$", re.DOTALL)
RE_UNION = re.compile(r"\\cup")
RE_BARE_COMMA_LIST = re.compile(r"^[^,()\[\]{}]+(?:,\s*[^,()\[\]{}]+)+$")
RE_THOUSANDS_SEP = re.compile(r",\d{3}(?:\D|$)")
RE_BASE_SUBSCRIPT = re.compile(
    r"^(?P<digits>[0-9A-Fa-f]+)_(?:\{)?(?P<base>[0-9]+)(?:\})?$"
)
RE_SYMPY_STYLE = re.compile(r"(?:sqrt|log|sin|cos|tan|exp)\s*\(|(?<!\^)\*\*")

SUPERSCRIPT_MAP = {
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "⁺": "+",
    "⁻": "-",
    "⁽": "(",
    "⁾": ")",
}
RE_SUPERSCRIPT_CHARS = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+")


def _convert_superscripts(text: str) -> str:
    if not text or not RE_SUPERSCRIPT_CHARS.search(text):
        return text

    def convert(s, base=None):
        converted = "".join(SUPERSCRIPT_MAP.get(ch, ch) for ch in s)
        return f"{base}**{converted}" if base else converted

    text = re.sub(
        r"([0-9A-Za-z\)\]\}])([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+)",
        lambda m: convert(m.group(2), base=m.group(1)),
        text,
    )
    text = convert(text)
    return text


def _strip_left_right(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\\left\s*", "", text)
    text = re.sub(r"\\right\s*", "", text)
    return text


def _strip_text_mbox_and_exponents(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\\text\s*\{[^}]*\}\s*\^\s*(\{[^}]*\}|\w+)", "", text)
    text = re.sub(r"\\mbox\s*\{[^}]*\}\s*\^\s*(\{[^}]*\}|\w+)", "", text)
    text = re.sub(r"\\text\s*\{[^}]*\}", "", text)
    text = re.sub(r"\\mbox\s*\{[^}]*\}", "", text)
    return text


def _insert_implicit_multiplication(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\\sqrt\s*([0-9]+)", r"\\sqrt{\1}", text)
    text = re.sub(r"(\d)\s*(\\sqrt)", r"\1\\cdot\2", text)
    text = re.sub(r"([\)\}])\s*(\\sqrt)", r"\1\\cdot\2", text)
    text = re.sub(r"(\d)\s*(\\pi)", r"\1\\cdot\2", text)
    text = re.sub(r"([\)\}])\s*(\\pi)", r"\1\\cdot\2", text)
    return text


def _preprocess(text: str) -> str:
    text = text.strip().strip("$")
    text = re.sub(r"\\\(|\\\)|\\\[|\\\]", "", text)
    text = _convert_superscripts(text)
    m = RE_LEADING_DOT.match(text)
    if m:
        text = "0" + text
    text = re.sub(r"\\left\s*", "", text)
    text = re.sub(r"\\right\s*", "", text)
    text = _strip_text_mbox_and_exponents(text)
    text = _insert_implicit_multiplication(text)
    text = re.sub(r"\^\s*\\circ", "", text)
    text = re.sub(r"\^\s*\{\s*\\circ\s*\}", "", text)
    text = text.replace("°", "")
    return text


def _parse_single_expr(text: str):
    if not text:
        return None
    text = _preprocess(text)
    if not text:
        return None
    return _latex_parse(text)


def _parse_text_answer(text: str):
    m = RE_TEXT.match(text.strip())
    if m:
        return m.group(1).strip()
    m = RE_MBOX.match(text.strip())
    if m:
        return m.group(1).strip()
    return None


def _parse_currency(text: str):
    m = RE_CURRENCY.match(text.strip())
    if m:
        num_str = re.sub(r"\\[,!]|\s|,", "", m.group(1))
        try:
            return float(num_str)
        except ValueError:
            return None
    return None


def _parse_interval_endpoint(text: str):
    text = text.strip()
    if text in (r"\infty", r"+\infty"):
        return oo
    if text == r"-\infty":
        return -oo
    return _parse_single_expr(text)


def _parse_interval(text: str):
    text = _strip_left_right(text.strip())
    m = RE_INTERVAL_BOUNDS.match(text)
    if not m:
        return None
    left_brace, left_val, right_val, right_brace = m.groups()
    a = _parse_interval_endpoint(left_val)
    b = _parse_interval_endpoint(right_val)
    if a is None or b is None:
        return None
    try:
        return Interval(
            a, b, left_open=(left_brace == "("), right_open=(right_brace == ")")
        )
    except Exception:
        return None


def _parse_union(text: str):
    parts = RE_UNION.split(text)
    if len(parts) < 2:
        return None
    intervals = []
    for p in parts:
        iv = _parse_interval(p.strip())
        if iv is None:
            return None
        intervals.append(iv)
    try:
        return Union(*intervals)
    except Exception:
        return None


def _parse_matrix(text: str):
    m = RE_MATRIX.search(text)
    if not m:
        return None
    inner = m.group(1).strip()
    rows_raw = re.split(r"\\\\", inner)
    rows = []
    for row in rows_raw:
        row = row.strip()
        if not row:
            continue
        cols = [c.strip() for c in row.split("&")]
        parsed_cols = []
        for c in cols:
            expr = _parse_single_expr(c)
            if expr is None:
                return None
            parsed_cols.append(expr)
        rows.append(parsed_cols)
    if not rows:
        return None
    try:
        return Matrix(rows)
    except Exception:
        return None


def _parse_tuple_or_list(text: str):
    text = _strip_left_right(text.strip())
    if not text or text[0] not in "([{" or text[-1] not in ")]}":
        return None
    inner = text[1:-1]
    parts = [p.strip() for p in inner.split(",")]
    if len(parts) < 2:
        return None
    parsed = []
    for p in parts:
        expr = _parse_single_expr(p)
        if expr is None:
            return None
        parsed.append(expr)
    return tuple(parsed)


def _parse_set_braces(text: str):
    m = re.match(r"^\\\{(.+)\\\}$", text.strip())
    if not m:
        return None
    inner = m.group(1)
    parts = [p.strip() for p in inner.split(",")]
    parsed = []
    for p in parts:
        if r"\pm" in p:
            m_pm = re.match(r"^(?P<a>.*)\\pm(?P<b>.*)$", p)
            if not m_pm:
                return None
            a, b = m_pm.group("a").strip(), m_pm.group("b").strip()
            for op in ("+", "-"):
                expr = _parse_single_expr(f"{a} {op} {b}")
                if expr is None:
                    return None
                parsed.append(expr)
        else:
            expr = _parse_single_expr(p)
            if expr is None:
                return None
            parsed.append(expr)
    try:
        return FiniteSet(*parsed)
    except Exception:
        return None


def _parse_plus_minus(text: str):
    m = RE_PM.match(text.strip())
    if not m:
        return None
    a = _parse_single_expr(m.group("a").strip())
    b = _parse_single_expr(m.group("b").strip())
    if a is None or b is None:
        return None
    try:
        return FiniteSet(a + b, a - b)
    except Exception:
        return None


def _parse_base_subscript(text: str):
    m = RE_BASE_SUBSCRIPT.match(text.strip())
    if not m:
        return None
    digits, base_str = m.group("digits"), m.group("base")
    try:
        base = int(base_str)
        if base < 2 or base > 36:
            return None
        int(digits, base)
        return f"{digits}_{base}"
    except ValueError:
        return None


def _parse_bare_comma_list(text: str):
    text = text.strip()
    if not RE_BARE_COMMA_LIST.match(text):
        return None
    clean = re.sub(r"\\[,!]|\s", "", text)
    if RE_THOUSANDS_SEP.search(clean):
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) < 2:
        return None
    parsed = []
    for p in parts:
        if not p:
            return None
        expr = _parse_single_expr(p)
        if expr is None:
            return None
        parsed.append(expr)
    return tuple(parsed)


def _parse_membership(text: str):
    m = RE_MEMBERSHIP.match(text.strip())
    if not m:
        return None
    lhs_raw, rhs_raw = m.group("lhs").strip(), m.group("rhs").strip()
    lhs = _parse_single_expr(lhs_raw) or (
        Symbol(lhs_raw) if re.fullmatch(r"[A-Za-z]", lhs_raw) else None
    )
    if lhs is None:
        return None
    rhs = (
        _parse_interval(rhs_raw)
        or _parse_set_braces(rhs_raw)
        or _parse_single_expr(rhs_raw)
    )
    if rhs is None:
        return None
    try:
        return Contains(lhs, rhs)
    except Exception:
        return None


def sympy_parser_hybrid(answer: str):
    """混合解析器主入口（原书 bonus/parser.py 同名函数移植）。

    按结构分派（\\in / \\pm / \\text / 货币 / 矩阵 / \\cup / 区间 /
    \\{集合\\} / 元组 / 进制下标 / 逗号列表 / sympy 风格表达式），
    返回 sympy 对象、str、tuple、float 之一；无法解析返回 None。
    """
    raw = answer.strip()
    if not raw:
        return None
    boxed = re.match(r"^\\boxed\{(.*)\}$", raw, re.DOTALL)
    if boxed:
        return sympy_parser_hybrid(boxed.group(1))
    text = _strip_left_right(raw)

    if r"\in" in text:
        result = _parse_membership(text)
        if result is not None:
            return result

    if r"\pm" in text:
        result = _parse_plus_minus(text)
        if result is not None:
            return result

    if RE_TEXT.match(raw) or RE_MBOX.match(raw):
        return _parse_text_answer(raw)

    if raw.startswith(r"\$"):
        result = _parse_currency(raw)
        if result is not None:
            return result

    if r"\begin{" in raw and "matrix" in raw:
        return _parse_matrix(raw)

    if r"\cup" in text:
        return _parse_union(text)

    if RE_INFTY.search(text) or (
        RE_TUPLE_LIKE.match(text) and ("]" in text or "[" in text)
    ):
        result = _parse_interval(text)
        if result is not None:
            return result

    if text.startswith(r"\{") and text.endswith(r"\}"):
        result = _parse_set_braces(text)
        if result is not None:
            return result

    if RE_TUPLE_LIKE.match(text):
        result = _parse_tuple_or_list(text)
        if result is not None:
            return result

    result = _parse_base_subscript(text)
    if result is not None:
        return result

    result = _parse_bare_comma_list(text)
    if result is not None:
        return result

    preprocessed = _preprocess(text)
    if not preprocessed:
        return None

    is_sympy_style = RE_SYMPY_STYLE.search(preprocessed) and "\\" not in preprocessed
    if is_sympy_style:
        try:
            return sympify(preprocessed)
        except Exception:
            pass

    # sympify 优先于 antlr LaTeX 解析：纯词/纯数字输入 sympify 语义更安全
    # （parse_latex 会把 SEVEN 拆成单字母符号乘积、2/3 返回未求值 Mul），
    # 含 LaTeX 命令（\frac 等）的输入 sympify 抛错后才落到 _latex_parse
    try:
        return sympify(preprocessed)
    except Exception:
        pass

    parsed = _latex_parse(preprocessed)
    if parsed is not None:
        return parsed
    return None


def normalize_text_hybrid(answer: str) -> str:
    """解析后的规范化字符串（原书 bonus/parser.py 同名函数移植）。"""
    if not answer:
        return ""

    expr = sympy_parser_hybrid(answer)

    if expr is None:
        return answer.strip()

    if isinstance(expr, str):
        return expr

    return str(expr)


# ------------------------------
# 语义等价比较与评分
# ------------------------------
def _text_equal(left, right):
    return str(left).strip().lower() == str(right).strip().lower()


def _is_set_like(x):
    from sympy.sets.sets import FiniteSet

    return isinstance(x, (set, frozenset)) or isinstance(x, FiniteSet)


def _semantic_equal(left, right):
    """sympy 对象/str/tuple/set 的统一语义等价判断。"""
    if _is_set_like(left) or _is_set_like(right):
        # 至少一侧是集合（含 sympy FiniteSet）：多重集语义，
        # 与逗号列表/元组无序等价（MATH 多答案 \{a,b\} 与 "a, b" 等价）
        try:
            seq_l, seq_r = list(left), list(right)
        except TypeError:
            return False
        if len(seq_l) != len(seq_r):
            return False
        remaining = list(seq_r)
        for a in seq_l:
            for i, b in enumerate(remaining):
                if _semantic_equal(a, b):
                    remaining.pop(i)
                    break
            else:
                return False
        return True
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(
            _semantic_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, str) or isinstance(right, str):
        return _text_equal(left, right)
    try:
        diff = simplify(left - right)
    except Exception:
        diff = None
    if diff is not None:
        if isinstance(diff, MatrixBase):
            return bool(diff == zeros(*diff.shape))
        return bool(diff == 0)
    return simplify(left) == simplify(right)


def parse_and_verify_answer(pred_text, gt_text):
    """advanced 评分入口：(pred, expected) -> bool。

    双方先经 sympy_parser_hybrid 解析，再按语义比较（数值化简、
    区间/集合/矩阵/元组结构相等）；任一侧解析失败时回退 basic
    grade_answer（normalize_text + sympy 数值等价），保证无 antlr
    环境下仍可用。
    """
    from inference.evaluation.benchmarks.math_verifiers import grade_answer

    if pred_text is None or gt_text is None:
        return False

    pred = sympy_parser_hybrid(pred_text)
    gt = sympy_parser_hybrid(gt_text)
    if pred is None or gt is None:
        return grade_answer(pred_text, gt_text)

    result = _semantic_equal(pred, gt)
    # hybrid 判 False 时再给 basic 一次机会（LaTeX 细节规范化差异）
    return result or grade_answer(pred_text, gt_text)


# ------------------------------
# test
# ------------------------------
# 测试代码 main 函数
def main():
    demos = [
        ("2/3", "4/6", True),
        ("0.5", "1/2", True),
        ("sqrt(2)", "2**(1/2)", True),
        (r"\text{42}", "42", True),
        (r"\boxed{(0, 1)}", "(0,1)", True),
        (r"\{1, 2\}", "2,1", True),
        ("1/2", "1/3", False),
    ]
    for pred, gt, expect in demos:
        got = parse_and_verify_answer(pred, gt)
        status = "PASS" if got == expect else "FAIL"
        print(f"{pred!r} vs {gt!r} -> {got} (expect {expect}) {status}")


if __name__ == "__main__":
    main()
