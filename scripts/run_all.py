from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行完整流程。")
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件路径。")
    parser.add_argument("--data-path", default=None, help="外部数据路径，支持 CSV 或 Excel。")
    parser.add_argument("--target", default=None, help="回归目标列。")
    parser.add_argument("--skip-traditional", action="store_true", help="跳过传统机器学习。")
    parser.add_argument("--skip-clustering", action="store_true", help="跳过聚类。")
    parser.add_argument("--skip-reduction", action="store_true", help="跳过降维。")
    parser.add_argument("--skip-explainability", action="store_true", help="跳过解释性分析。")
    parser.add_argument("--include-deep-learning", action="store_true", help="运行 PyTorch 深度学习流程。")
    parser.add_argument("--skip-shap", action="store_true", help="解释性分析中跳过 SHAP。")
    parser.add_argument("--no-xgboost", action="store_true", help="传统机器学习中不使用 XGBoost。")
    return parser.parse_args()


def run_command(args: list[str]) -> None:
    command = [sys.executable] + args

    print("=" * 80)
    print("运行命令:")
    print(" ".join(command))
    print("=" * 80)

    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()

    common = ["--config", args.config]

    if args.data_path:
        common += ["--data-path", args.data_path]

    if not args.data_path:
        run_command(["scripts/make_dataset.py", "--config", args.config])

    run_command(["scripts/run_data_check.py", *common])

    run_command(["scripts/run_feature_engineering.py", *common])

    if not args.skip_traditional:
        traditional_args = ["scripts/run_traditional_ml.py", *common]

        if args.target:
            traditional_args += ["--regression-target", args.target]

        if args.no_xgboost:
            traditional_args += ["--no-xgboost"]

        run_command(traditional_args)

    if not args.skip_clustering:
        run_command(["scripts/run_clustering.py", *common])

    if not args.skip_reduction:
        run_command(["scripts/run_dimensionality_reduction.py", *common])

    if not args.skip_explainability:
        explain_args = ["scripts/run_shap_analysis.py", *common]

        if args.target:
            explain_args += ["--target", args.target]

        if args.skip_shap:
            explain_args += ["--skip-shap"]

        run_command(explain_args)

    if args.include_deep_learning:
        run_command(["scripts/run_deep_learning.py", *common])

    run_command(["scripts/run_evaluation.py", "--config", args.config])


if __name__ == "__main__":
    main()