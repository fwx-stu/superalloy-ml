from pathlib import Path

import pandas as pd
import pytest

from superalloy_ml.data import generate_synthetic_superalloy_dataset, split_features_targets
from superalloy_ml.explainability import (
    compute_permutation_importance,
    compute_shap_summary,
    get_builtin_feature_importance,
    get_feature_names,
    plot_importance_bar,
    save_explainability_tables,
    transform_features_for_model,
)
from superalloy_ml.traditional_ml import build_pipeline, get_regression_models


def _trained_tree_pipeline():
    df = generate_synthetic_superalloy_dataset(n_samples=120, random_state=42)
    X, y = split_features_targets(df, target="creep_log_life_h")

    model = get_regression_models(include_xgboost=False)["random_forest_regressor"]
    pipeline = build_pipeline(X, model=model)
    pipeline.fit(X, y)

    return pipeline, X, y


def test_get_feature_names():
    pipeline, X, _ = _trained_tree_pipeline()

    names = get_feature_names(pipeline, X)

    assert isinstance(names, list)
    assert len(names) > 0


def test_transform_features_for_model():
    pipeline, X, _ = _trained_tree_pipeline()

    X_transformed = transform_features_for_model(pipeline, X.head(10))

    assert X_transformed.shape[0] == 10


def test_builtin_feature_importance():
    pipeline, X, _ = _trained_tree_pipeline()

    table = get_builtin_feature_importance(pipeline, X)

    assert isinstance(table, pd.DataFrame)
    assert not table.empty
    assert "feature" in table.columns
    assert "importance" in table.columns


def test_permutation_importance():
    pipeline, X, y = _trained_tree_pipeline()

    table = compute_permutation_importance(
        model=pipeline,
        X=X.head(40),
        y=y.head(40),
        scoring="neg_root_mean_squared_error",
        n_repeats=2,
        random_state=42,
    )

    assert isinstance(table, pd.DataFrame)
    assert not table.empty
    assert "feature" in table.columns
    assert "importance_mean" in table.columns


def test_plot_importance_bar(tmp_path: Path):
    table = pd.DataFrame(
        {
            "feature": ["a", "b", "c"],
            "importance": [0.3, 0.2, 0.1],
        }
    )

    path = tmp_path / "importance.png"

    plot_importance_bar(
        table=table,
        value_column="importance",
        path=path,
        title="特征重要性",
    )

    assert path.exists()


def test_save_explainability_tables(tmp_path: Path):
    table = pd.DataFrame(
        {
            "feature": ["a", "b"],
            "importance": [0.2, 0.1],
        }
    )

    paths = save_explainability_tables(
        output_dir=tmp_path,
        builtin_table=table,
    )

    assert "builtin" in paths
    assert paths["builtin"].exists()


def test_compute_shap_summary_optional():
    pytest.importorskip("shap")

    pipeline, X, _ = _trained_tree_pipeline()

    table = compute_shap_summary(
        model=pipeline,
        X=X.head(20),
        max_samples=20,
    )

    assert isinstance(table, pd.DataFrame)
    assert not table.empty
    assert "feature" in table.columns
    assert "mean_abs_shap" in table.columns