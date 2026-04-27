import numpy as np

from superalloy_ml.evaluation import classification_metrics, regression_metrics, rmse


def test_rmse():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 5.0])

    value = rmse(y_true, y_pred)

    assert value > 0


def test_regression_metrics():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])

    metrics = regression_metrics(y_true, y_pred)

    assert "mae" in metrics
    assert "mse" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert "mape" in metrics
    assert "max_error" in metrics


def test_classification_metrics():
    y_true = np.array([0, 1, 1, 2])
    y_pred = np.array([0, 1, 0, 2])

    metrics = classification_metrics(y_true, y_pred)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics

    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["f1"] <= 1