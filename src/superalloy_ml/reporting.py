from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from superalloy_ml.utils import ensure_dir, save_json


def create_output_dirs(base_dir: str | Path = "outputs") -> dict[str, Path]:
    """
    创建统一输出目录。
    """
    base_dir = Path(base_dir)

    dirs = {
        "base": ensure_dir(base_dir),
        "models": ensure_dir(base_dir / "models"),
        "reports": ensure_dir(base_dir / "reports"),
        "figures": ensure_dir(base_dir / "figures"),
        "predictions": ensure_dir(base_dir / "predictions"),
    }

    return dirs


def save_table(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    """
    保存表格。
    """
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=index)


def save_metrics(metrics: dict, path: str | Path) -> None:
    """
    保存指标。
    """
    save_json(metrics, path)


def plot_model_comparison(
    results: pd.DataFrame,
    metric: str,
    path: str | Path,
    title: str,
    top_k: int = 12,
    higher_is_better: bool = True,
) -> None:
    """
    绘制模型对比图。
    """
    path = Path(path)
    ensure_dir(path.parent)

    if metric not in results.columns:
        raise ValueError(f"结果表中不存在指标列: {metric}")

    plot_df = results.copy()
    plot_df = plot_df.sort_values(metric, ascending=not higher_is_better).head(top_k)

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["model_name"], plot_df[metric])
    plt.xlabel(metric)
    plt.ylabel("模型")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_prediction_vs_actual(
    y_true,
    y_pred,
    path: str | Path,
    title: str = "预测值与真实值对比",
) -> None:
    """
    绘制预测值与真实值对比图。
    """
    path = Path(path)
    ensure_dir(path.parent)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)

    min_value = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_value = max(float(np.max(y_true)), float(np.max(y_pred)))

    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")
    plt.xlabel("真实值")
    plt.ylabel("预测值")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_residuals(
    y_true,
    y_pred,
    path: str | Path,
    title: str = "残差图",
) -> None:
    """
    绘制残差图。
    """
    path = Path(path)
    ensure_dir(path.parent)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred

    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(0, linestyle="--")
    plt.xlabel("预测值")
    plt.ylabel("残差")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_confusion_matrix(
    y_true,
    y_pred,
    path: str | Path,
    title: str = "混淆矩阵",
) -> None:
    """
    绘制混淆矩阵。
    """
    path = Path(path)
    ensure_dir(path.parent)

    cm = confusion_matrix(y_true, y_pred)

    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    display.plot(values_format="d")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def write_markdown_report(
    path: str | Path,
    title: str,
    sections: dict[str, str],
) -> None:
    """
    生成 Markdown 报告。
    """
    path = Path(path)
    ensure_dir(path.parent)

    lines = [f"# {title}", ""]

    for section_title, content in sections.items():
        lines.append(f"## {section_title}")
        lines.append("")
        lines.append(content)
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")