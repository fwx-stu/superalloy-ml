from pathlib import Path

import pandas as pd

from superalloy_ml.data import generate_synthetic_superalloy_dataset, split_features_targets
from superalloy_ml.traditional_ml import (
    build_pipeline,
    cross_validate_models,
    get_best_model_name,
    get_classification_models,
    get_regression_models,
    load_model,
    save_model,
    train_test_evaluate_models,
)


def test_get_regression_models():
    models = get_regression_models(include_xgboost=False)

    expected = [
        "linear_regression",
        "ridge",
        "lasso",
        "elasticnet",
        "knn_regressor",
        "svr",
        "decision_tree_regressor",
        "random_forest_regressor",
        "extra_trees_regressor",
        "gradient_boosting_regressor",
    ]

    for name in expected:
        assert name in models


def test_get_classification_models():
    models = get_classification_models(include_xgboost=False)

    expected = [
        "logistic_regression",
        "knn_classifier",
        "svm_classifier",
        "decision_tree_classifier",
        "random_forest_classifier",
        "extra_trees_classifier",
        "gradient_boosting_classifier",
    ]

    for name in expected:
        assert name in models


def test_build_pipeline_regression():
    df = generate_synthetic_superalloy_dataset(n_samples=80, random_state=42)
    X, y = split_features_targets(df, target="creep_log_life_h")

    model = get_regression_models(include_xgboost=False)["ridge"]
    pipeline = build_pipeline(X, model=model)

    pipeline.fit(X, y)
    preds = pipeline.predict(X.head(5))

    assert len(preds) == 5


def test_train_test_evaluate_regression():
    df = generate_synthetic_superalloy_dataset(n_samples=120, random_state=42)
    X, y = split_features_targets(df, target="creep_log_life_h")

    results, trained_models = train_test_evaluate_models(
        X=X,
        y=y,
        task="regression",
        test_size=0.25,
        random_state=42,
        include_xgboost=False,
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert "model_name" in results.columns
    assert "rmse" in results.columns
    assert "r2" in results.columns
    assert len(trained_models) == len(results)

    best_model_name = get_best_model_name(results, task="regression")
    assert best_model_name in trained_models


def test_train_test_evaluate_classification():
    df = generate_synthetic_superalloy_dataset(n_samples=150, random_state=42)
    X, y = split_features_targets(df, target="creep_life_class")

    results, trained_models = train_test_evaluate_models(
        X=X,
        y=y,
        task="classification",
        test_size=0.25,
        random_state=42,
        include_xgboost=False,
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert "model_name" in results.columns
    assert "accuracy" in results.columns
    assert "f1" in results.columns
    assert len(trained_models) == len(results)

    best_model_name = get_best_model_name(results, task="classification")
    assert best_model_name in trained_models


def test_cross_validate_regression():
    df = generate_synthetic_superalloy_dataset(n_samples=120, random_state=42)
    X, y = split_features_targets(df, target="creep_log_life_h")

    results = cross_validate_models(
        X=X,
        y=y,
        task="regression",
        cv=3,
        random_state=42,
        include_xgboost=False,
    )

    assert len(results) > 0
    assert "cv_mean" in results.columns
    assert "cv_std" in results.columns


def test_cross_validate_classification():
    df = generate_synthetic_superalloy_dataset(n_samples=150, random_state=42)
    X, y = split_features_targets(df, target="creep_life_class")

    results = cross_validate_models(
        X=X,
        y=y,
        task="classification",
        cv=3,
        random_state=42,
        include_xgboost=False,
    )

    assert len(results) > 0
    assert "cv_mean" in results.columns
    assert "cv_std" in results.columns


def test_save_and_load_model(tmp_path: Path):
    df = generate_synthetic_superalloy_dataset(n_samples=80, random_state=42)
    X, y = split_features_targets(df, target="creep_log_life_h")

    model = get_regression_models(include_xgboost=False)["ridge"]
    pipeline = build_pipeline(X, model=model)
    pipeline.fit(X, y)

    model_path = tmp_path / "ridge_model.joblib"
    save_model(pipeline, model_path)

    loaded_model = load_model(model_path)
    preds = loaded_model.predict(X.head(3))

    assert len(preds) == 3