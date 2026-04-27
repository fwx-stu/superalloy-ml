from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    "requirements-dl.txt",
    "requirements-optional.txt",
    "pyproject.toml",
    "Makefile",
    ".gitignore",
    "configs/default.yaml",
    "data/README.md",
    "scripts/make_dataset.py",
    "scripts/run_data_check.py",
    "scripts/run_feature_engineering.py",
    "scripts/run_traditional_ml.py",
    "scripts/run_clustering.py",
    "scripts/run_dimensionality_reduction.py",
    "scripts/run_deep_learning.py",
    "scripts/run_shap_analysis.py",
    "scripts/run_evaluation.py",
    "scripts/run_all.py",
    "src/superalloy_ml/__init__.py",
    "src/superalloy_ml/constants.py",
    "src/superalloy_ml/utils.py",
    "src/superalloy_ml/validation.py",
    "src/superalloy_ml/data.py",
    "src/superalloy_ml/preprocessing.py",
    "src/superalloy_ml/features.py",
    "src/superalloy_ml/traditional_ml.py",
    "src/superalloy_ml/clustering.py",
    "src/superalloy_ml/dimensionality_reduction.py",
    "src/superalloy_ml/explainability.py",
    "src/superalloy_ml/evaluation.py",
    "src/superalloy_ml/reporting.py",
    "src/superalloy_ml/deep_learning/__init__.py",
    "src/superalloy_ml/deep_learning/datasets.py",
    "src/superalloy_ml/deep_learning/mlp.py",
    "src/superalloy_ml/deep_learning/cnn.py",
    "src/superalloy_ml/deep_learning/rnn_lstm.py",
    "src/superalloy_ml/deep_learning/transformer.py",
    "src/superalloy_ml/deep_learning/trainer.py",
    "docs/项目说明.md",
    "docs/数据字段说明.md",
    "docs/特征工程说明.md",
    "docs/算法说明.md",
    "docs/评估指标说明.md",
    "docs/AI辅助说明.md",
]


REQUIRED_DIRS = [
    "configs",
    "data/raw",
    "data/processed",
    "scripts",
    "src/superalloy_ml",
    "src/superalloy_ml/deep_learning",
    "notebooks",
    "docs",
    "tests",
    "outputs/models",
    "outputs/reports",
    "outputs/figures",
    "outputs/predictions",
]


def main() -> None:
    missing_files = []
    missing_dirs = []

    for relative_path in REQUIRED_FILES:
        path = ROOT / relative_path
        if not path.exists():
            missing_files.append(relative_path)

    for relative_path in REQUIRED_DIRS:
        path = ROOT / relative_path
        if not path.exists():
            missing_dirs.append(relative_path)

    if missing_dirs:
        print("缺少目录:")
        for path in missing_dirs:
            print(f"  - {path}")

    if missing_files:
        print("缺少文件:")
        for path in missing_files:
            print(f"  - {path}")

    if missing_dirs or missing_files:
        sys.exit(1)

    print("项目结构检查通过。")


if __name__ == "__main__":
    main()