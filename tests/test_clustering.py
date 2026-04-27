import numpy as np
import pandas as pd

from superalloy_ml.clustering import (
    calculate_clustering_metrics,
    make_cluster_assignment_table,
    make_cluster_embedding_table,
    prepare_clustering_matrix,
    run_agglomerative,
    run_all_clustering,
    run_dbscan,
    run_kmeans,
)
from superalloy_ml.constants import TARGET_COLUMNS
from superalloy_ml.data import generate_synthetic_superalloy_dataset
from superalloy_ml.features import get_numeric_feature_columns


def _get_sample_X(n_samples=80):
    df = generate_synthetic_superalloy_dataset(n_samples=n_samples, random_state=42)
    columns = get_numeric_feature_columns(df, TARGET_COLUMNS)
    return df[columns]


def test_prepare_clustering_matrix():
    X = _get_sample_X(50)
    X_processed, preprocessor = prepare_clustering_matrix(X)

    assert X_processed.shape[0] == X.shape[0]
    assert preprocessor is not None


def test_calculate_clustering_metrics():
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
            [5.1, 5.0],
        ]
    )
    labels = np.array([0, 0, 1, 1])

    metrics = calculate_clustering_metrics(X, labels)

    assert metrics["n_clusters"] == 2
    assert metrics["n_noise"] == 0


def test_run_kmeans():
    X = _get_sample_X(60)
    X_processed, _ = prepare_clustering_matrix(X)

    result = run_kmeans(X_processed, n_clusters=3, random_state=42)

    assert result.algorithm == "kmeans"
    assert len(result.labels) == X.shape[0]
    assert result.embedding_2d.shape == (X.shape[0], 2)


def test_run_dbscan():
    X = _get_sample_X(60)
    X_processed, _ = prepare_clustering_matrix(X)

    result = run_dbscan(X_processed, eps=5.0, min_samples=3)

    assert result.algorithm == "dbscan"
    assert len(result.labels) == X.shape[0]
    assert result.embedding_2d.shape == (X.shape[0], 2)


def test_run_agglomerative():
    X = _get_sample_X(60)
    X_processed, _ = prepare_clustering_matrix(X)

    result = run_agglomerative(X_processed, n_clusters=3)

    assert result.algorithm == "agglomerative"
    assert len(result.labels) == X.shape[0]
    assert result.embedding_2d.shape == (X.shape[0], 2)


def test_run_all_clustering():
    X = _get_sample_X(60)

    summary, results, X_processed = run_all_clustering(
        X=X,
        n_clusters=3,
        dbscan_eps=5.0,
        dbscan_min_samples=3,
    )

    assert isinstance(summary, pd.DataFrame)
    assert "algorithm" in summary.columns
    assert "kmeans" in results
    assert "dbscan" in results
    assert "agglomerative" in results
    assert X_processed.shape[0] == X.shape[0]


def test_cluster_assignment_table():
    X = _get_sample_X(60)
    _, results, _ = run_all_clustering(
        X=X,
        n_clusters=3,
        dbscan_eps=5.0,
        dbscan_min_samples=3,
    )

    table = make_cluster_assignment_table(results)

    assert table.shape[0] == X.shape[0]
    assert "kmeans_label" in table.columns


def test_cluster_embedding_table():
    X = _get_sample_X(60)
    X_processed, _ = prepare_clustering_matrix(X)

    result = run_kmeans(X_processed, n_clusters=3, random_state=42)
    table = make_cluster_embedding_table(result)

    assert table.shape[0] == X.shape[0]
    assert {"x", "y", "cluster"}.issubset(table.columns)