from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from superalloy_ml.reporting import create_output_dirs, write_markdown_report
from superalloy_ml.utils import get_config, print_section


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总模型评估结果。")
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件路径。")
    return parser.parse_args()


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _format_table_for_markdown(df: pd.DataFrame | None, max_rows: int = 10) -> str:
    if df is None or df.empty:
        return "暂无结果文件。"

    return df.head(max_rows).to_markdown(index=False)


def _select_best_regression_model(df: pd.DataFrame | None) -> dict | None:
    if df is None or df.empty:
        return None

    if "r2" in df.columns:
        sorted_df = df.sort_values(["r2", "rmse"], ascending=[False, True])
    elif "rmse" in df.columns:
        sorted_df = df.sort_values("rmse", ascending=True)
    else:
        sorted_df = df.copy()

    row = sorted_df.iloc[0].to_dict()
    return row


def _select_best_classification_model(df: pd.DataFrame | None) -> dict | None:
    if df is None or df.empty:
        return None

    if "f1" in df.columns:
        sorted_df = df.sort_values(["f1", "accuracy"], ascending=[False, False])
    elif "accuracy" in df.columns:
        sorted_df = df.sort_values("accuracy", ascending=False)
    else:
        sorted_df = df.copy()

    row = sorted_df.iloc[0].to_dict()
    return row


def _safe_json_value(value):
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        return value.item()

    return value


def _clean_dict(obj: dict | None) -> dict | None:
    if obj is None:
        return None

    return {key: _safe_json_value(value) for key, value in obj.items()}


def main() -> None:
    args = parse_args()
    config = get_config(args.config)

    print_section("汇总评估结果")

    output_base_dir = config.get("outputs", {}).get("base_dir", "outputs")
    output_dirs = create_output_dirs(ROOT / output_base_dir)

    report_dir = output_dirs["reports"]

    regression_test = _read_csv_if_exists(report_dir / "traditional_regression_test_results.csv")
    regression_cv = _read_csv_if_exists(report_dir / "traditional_regression_cv_results.csv")
    classification_test = _read_csv_if_exists(
        report_dir / "traditional_classification_test_results.csv"
    )
    classification_cv = _read_csv_if_exists(
        report_dir / "traditional_classification_cv_results.csv"
    )
    clustering_summary = _read_csv_if_exists(report_dir / "clustering_summary.csv")
    reduction_summary = _read_csv_if_exists(report_dir / "dimensionality_reduction_summary.csv")
    deep_learning_results = _read_csv_if_exists(report_dir / "deep_learning_results.csv")

    best_regression = _clean_dict(_select_best_regression_model(regression_test))
    best_classification = _clean_dict(_select_best_classification_model(classification_test))

    summary = {
        "best_regression_model": best_regression,
        "best_classification_model": best_classification,
        "available_reports": {
            "traditional_regression_test_results": regression_test is not None,
            "traditional_regression_cv_results": regression_cv is not None,
            "traditional_classification_test_results": classification_test is not None,
            "traditional_classification_cv_results": classification_cv is not None,
            "clustering_summary": clustering_summary is not None,
            "dimensionality_reduction_summary": reduction_summary is not None,
            "deep_learning_results": deep_learning_results is not None,
        },
    }

    summary_json_path = report_dir / "summary.json"
    summary_json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    sections = {
        "整体说明": (
            "本报告汇总项目运行过程中生成的主要结果。"
            "如果某一部分显示暂无结果文件，说明对应脚本尚未运行，或该流程被跳过。"
        ),
        "最佳回归模型": (
            "暂无回归结果。"
            if best_regression is None
            else "\n".join([f"- {key}: {value}" for key, value in best_regression.items()])
        ),
        "最佳分类模型": (
            "暂无分类结果。"
            if best_classification is None
            else "\n".join([f"- {key}: {value}" for key, value in best_classification.items()])
        ),
        "传统机器学习回归结果": _format_table_for_markdown(regression_test),
        "传统机器学习回归交叉验证": _format_table_for_markdown(regression_cv),
        "传统机器学习分类结果": _format_table_for_markdown(classification_test),
        "传统机器学习分类交叉验证": _format_table_for_markdown(classification_cv),
        "聚类结果": _format_table_for_markdown(clustering_summary),
        "降维结果": _format_table_for_markdown(reduction_summary),
        "深度学习结果": _format_table_for_markdown(deep_learning_results),
        "输出目录": (
            f"- 模型文件: `{output_dirs['models']}`\n"
            f"- 结果表: `{output_dirs['reports']}`\n"
            f"- 图像文件: `{output_dirs['figures']}`\n"
            f"- 预测结果: `{output_dirs['predictions']}`"
        ),
    }

    report_path = report_dir / "summary_report.md"
    write_markdown_report(
        path=report_path,
        title="模型结果汇总报告",
        sections=sections,
    )

    print(f"汇总 JSON: {summary_json_path}")
    print(f"汇总报告: {report_path}")


if __name__ == "__main__":
    main()