# -*- coding: utf-8 -*-

# ***************************************************
# * File        : tracking.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102421
# * Description : RL 训练指标跟踪（reasoning-from-scratch ch06/ch07 的 metrics log）。
# *               MetricsTracker：动态列指标收集 -> CSV -> 摘要统计；
# *               compute_clip_fraction：clip 分支生效占比；
# *               plot_metrics：CSV 折线图（matplotlib 懒导入）。
# * Link        : https://github.com/rasbt/reasoning-from-scratch/tree/main/ch07/03_rlvr_grpo_scripts_advanced
# * Requirement : matplotlib（仅 plot_metrics 需要）
# ***************************************************

__all__ = [
    "MetricsTracker",
    "compute_clip_fraction",
    "plot_metrics",
]

# python libraries
import csv
from pathlib import Path

import torch

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class MetricsTracker:
    """训练指标收集器：``append(**fields)`` 动态列，``to_csv`` 导出，``summary`` 汇总。"""

    def __init__(self):
        self.rows = []

    def __len__(self):
        return len(self.rows)

    @property
    def columns(self):
        """按首次出现顺序合并所有行的字段名。"""
        cols = []
        for row in self.rows:
            for key in row:
                if key not in cols:
                    cols.append(key)
        return cols

    def append(self, **fields):
        """追加一条指标记录（各行列可不一致，缺项在 CSV 中留空）。"""
        self.rows.append(dict(fields))

    def to_csv(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns, restval="")
            writer.writeheader()
            writer.writerows(self.rows)

    def summary(self):
        """每列的 mean/min/max/last（非数值列不参与统计）。"""
        stats = {}
        for col in self.columns:
            values = [
                row[col] for row in self.rows
                if isinstance(row.get(col), (int, float))
                and not isinstance(row.get(col), bool)
            ]
            if values:
                stats[col] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "last": values[-1],
                }
        return stats


def compute_clip_fraction(ratio, advantages, clip_eps, clip_eps_high=None) -> float:
    """``min(ratio·A, clip(ratio)·A)`` 中 clip 分支生效的 token 占比。

    clip 生效与优势符号相关：ratio 越上界（1 + clip_eps_high，None 时
    对称 1 + clip_eps）且 A>0，或 ratio 破下界（1 - clip_eps）且 A<0。

    Args:
        ratio: (G, L) 重要性采样比率
        advantages: (G,) 或 (G, L) 优势
    """
    low = 1.0 - clip_eps
    high = 1.0 + (clip_eps if clip_eps_high is None else clip_eps_high)
    adv = advantages if advantages.dim() == ratio.dim() else advantages.unsqueeze(1)
    active = ((ratio > high) & (adv > 0)) | ((ratio < low) & (adv < 0))
    return float(active.float().mean().item())


def plot_metrics(csv_path, out_png):
    """把 MetricsTracker 写出的 CSV 各数值列画成折线图并保存为 PNG。

    x 轴取 "step" 列（存在且为数值时），否则用行号。
    """
    try:
        import matplotlib
    except ImportError as exc:
        raise ImportError(
            "plot_metrics 需要 matplotlib，当前环境未安装；"
            "请先 `pip install matplotlib`"
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_path = Path(csv_path)
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"{csv_path} 无数据行，无法绘图")

    def _is_float(value):
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False

    numeric_cols = [
        c for c in fieldnames
        if c != "step" and all(_is_float(row.get(c)) for row in rows)
    ]
    if not numeric_cols:
        raise ValueError(f"{csv_path} 无数值列，无法绘图")

    if "step" in fieldnames and all(_is_float(row.get("step")) for row in rows):
        x = [float(row["step"]) for row in rows]
        xlabel = "step"
    else:
        x = list(range(len(rows)))
        xlabel = "row"

    fig, ax = plt.subplots(figsize=(8, 5))
    for col in numeric_cols:
        ax.plot(x, [float(row[col]) for row in rows], label=col)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("value")
    ax.legend()
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
