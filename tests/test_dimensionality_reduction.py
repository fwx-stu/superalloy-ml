import pandas as pd

from superalloy_ml.constants import TARGET_COLUMNS
from superalloy_ml.data import generate_synthetic_superalloy_dataset
from superalloy_ml.dimensionality_reduction import (
    make_embedding_table,
    prepare_reduction_matrix,
    run_all_reductions,
    run_pca,
    run_tsne,
)
from superalloy_ml.features import get_numeric_feature_columns


def _get_sample_X(n_samples=80):
    df = generate_synthetic_superalloy_dataset(n_samples=n_samples, random_state=42)
    columns = get_numeric_feature_columns(df, TARGET_COLUMNS)
    return df[columns], df


def test_prepare_reduction_matrix():
    X, _ = _get_sample_X(50)
    X_processed, preprocessor = prepare_reduction_matrix(X)

    assert X_processed.shape[0] == X.shape[0]
    assert preprocessor is not None


def test_run_pca():
    X, _ = _get_sample_X(60)
    X_processed, _ = prepare_reduction_matrix(X)

    result = run_pca(X_processed, n_components=2, random_state=42)

    assert result.algorithm == "pca"
    assert result.embedding.shape == (X.shape[0], 2)
    assert "explained_variance_ratio" in result.metadata


def test_run_tsne():
    X, _ = _get_sample_X(40)
    X_processed, _ = prepare_reduction_matrix(X)

    result = run_tsne(
        X_processed,
        n_components=2,
        perplexity=10.0,
        random_state=42,
    )

    assert result.algorithm == "tsne"
    assert result.embedding.shape == (X.shape[0], 2)
    assert "kl_divergence" in result.metadata


def test_run_all_reductions():
    X, _ = _get_sample_X(50)

    summary, results, X_processed = run_all_reductions(
        X=X,
        tsne_perplexity=10.0,
        random_state=42,
    )

    assert isinstance(summary, pd.DataFrame)
    assert "algorithm" in summary.columns
    assert "pca" in results
    assert "tsne" in results
    assert X_processed.shape[0] == X.shape[0]


def test_make_embedding_table():
    X, df = _get_sample_X(50)
    X_processed, _ = prepare_reduction_matrix(X)

    result = run_pca(X_processed, n_components=2, random_state=42)
    table = make_embedding_table(result, label=df["creep_life_class"])

    assert table.shape[0] == X.shape[0]
    assert {"x", "y", "label"}.issubset(table.columns)