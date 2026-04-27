from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from superalloy_ml.preprocessing import create_preprocessor


@dataclass
class ReductionResult:
    algorithm: str
    embedding: np.ndarray
    model: Any
    metadata: dict[str, Any]


def _to_dense_array(X) -> np.ndarray:
    """
    将矩阵转换为 numpy 数组。
    """
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)


def prepare_reduction_matrix(
    X: pd.DataFrame,
    scaler: str = "standard",
) -> tuple[np.ndarray, Any]:
    """
    构建降维输入矩阵。
    """
    preprocessor = create_preprocessor(X, scaler=scaler)
    X_processed = preprocessor.fit_transform(X)
    X_processed = _to_dense_array(X_processed)

    return X_processed, preprocessor


def run_pca(
    X: np.ndarray,
    n_components: int = 2,
    random_state: int = 42,
) -> ReductionResult:
    """
    运行 PCA 降维。
    """
    n_components = min(n_components, X.shape[0], X.shape[1])

    model = PCA(n_components=n_components, random_state=random_state)
    embedding = model.fit_transform(X)

    metadata = {
        "n_components": int(n_components),
        "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
        "explained_variance_ratio_sum": float(np.sum(model.explained_variance_ratio_)),
    }

    return ReductionResult(
        algorithm="pca",
        embedding=embedding,
        model=model,
        metadata=metadata,
    )


def run_tsne(
    X: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    learning_rate: str | float = "auto",
    random_state: int = 42,
) -> ReductionResult:
    """
    运行 t-SNE 降维。

    t-SNE 对样本数量和 perplexity 较敏感，这里会自动限制 perplexity。
    """
    n_samples = X.shape[0]

    if n_samples <= 3:
        raise ValueError("t-SNE 至少需要 4 个样本。")

    max_perplexity = max(2.0, (n_samples - 1) / 3)
    safe_perplexity = min(float(perplexity), max_perplexity)

    model = TSNE(
        n_components=n_components,
        perplexity=safe_perplexity,
        learning_rate=learning_rate,
        init="pca",
        random_state=random_state,
    )
    embedding = model.fit_transform(X)

    metadata = {
        "n_components": int(n_components),
        "perplexity": float(safe_perplexity),
        "kl_divergence": float(model.kl_divergence_),
    }

    return ReductionResult(
        algorithm="tsne",
        embedding=embedding,
        model=model,
        metadata=metadata,
    )


def run_all_reductions(
    X: pd.DataFrame,
    scaler: str = "standard",
    random_state: int = 42,
    tsne_perplexity: float = 30.0,
) -> tuple[pd.DataFrame, dict[str, ReductionResult], np.ndarray]:
    """
    运行 PCA 和 t-SNE。
    """
    X_processed, _ = prepare_reduction_matrix(X, scaler=scaler)

    results = {
        "pca": run_pca(
            X_processed,
            n_components=2,
            random_state=random_state,
        ),
        "tsne": run_tsne(
            X_processed,
            n_components=2,
            perplexity=tsne_perplexity,
            random_state=random_state,
        ),
    }

    rows = []
    for name, result in results.items():
        row = {
            "algorithm": name,
            "n_components": int(result.embedding.shape[1]),
        }
        row.update(result.metadata)
        rows.append(row)

    summary = pd.DataFrame(rows)

    return summary, results, X_processed


def make_embedding_table(
    result: ReductionResult,
    label=None,
) -> pd.DataFrame:
    """
    生成降维结果表。
    """
    if result.embedding.shape[1] < 2:
        raise ValueError("可视化至少需要二维坐标。")

    table = pd.DataFrame(
        {
            "x": result.embedding[:, 0],
            "y": result.embedding[:, 1],
        }
    )

    if label is not None:
        table["label"] = np.asarray(label)

    return table