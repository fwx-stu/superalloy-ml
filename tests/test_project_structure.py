from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_core_files_exist():
    required_files = [
        "README.md",
        "requirements.txt",
        "requirements-dl.txt",
        "requirements-optional.txt",
        "pyproject.toml",
        "Makefile",
        ".gitignore",
        "configs/default.yaml",
        "data/README.md",
        "src/superalloy_ml/__init__.py",
        "src/superalloy_ml/constants.py",
        "src/superalloy_ml/utils.py",
        "src/superalloy_ml/data.py",
        "src/superalloy_ml/features.py",
        "src/superalloy_ml/preprocessing.py",
        "src/superalloy_ml/validation.py",
        "src/superalloy_ml/evaluation.py",
        "src/superalloy_ml/reporting.py",
        "src/superalloy_ml/traditional_ml.py",
        "src/superalloy_ml/clustering.py",
        "src/superalloy_ml/dimensionality_reduction.py",
        "src/superalloy_ml/explainability.py",
    ]

    for relative_path in required_files:
        assert (ROOT / relative_path).exists(), relative_path


def test_required_script_files_exist():
    required_files = [
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
        "scripts/check_project_structure.py",
    ]

    for relative_path in required_files:
        assert (ROOT / relative_path).exists(), relative_path


def test_required_output_dirs_exist():
    required_dirs = [
        "outputs/models",
        "outputs/reports",
        "outputs/figures",
        "outputs/predictions",
    ]

    for relative_path in required_dirs:
        assert (ROOT / relative_path).exists(), relative_path


def test_required_docs_exist():
    required_files = [
        "docs/项目说明.md",
        "docs/数据字段说明.md",
        "docs/特征工程说明.md",
        "docs/算法说明.md",
        "docs/评估指标说明.md",
        "docs/AI辅助说明.md",
    ]

    for relative_path in required_files:
        assert (ROOT / relative_path).exists(), relative_path