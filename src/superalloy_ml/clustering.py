from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from superalloy_ml.preprocessing import create_preprocessor


@dataclass
class ClusteringResult:
    algorithm: str
    labels: np.ndarray
    metrics: dict[str, float | int | None]
    model: Any
    embedding_2d: np.ndarray | None = None


def _to_dense_array(X) -> np.ndarray:
    """
    将预处理后的矩阵转换为 numpy 数组。
    """
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)


def prepare_clustering_matrix(
    X: pd.DataFrame,
    scaler: str = "standard",
) -> tuple[np.ndarray, Any]:
    """
    构建聚类输入矩阵。

    聚类对特征尺度敏感，因此默认使用标准化。
    """
    preprocessor = create_preprocessor(X, scaler=scaler)
    X_processed = preprocessor.fit_transform(X)
    X_processed = _to_dense_array(X_processed)

    return X_processed, preprocessor


def calculate_clustering_metrics(
    X: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float | int | None]:
    """
    计算聚类评估指标。

    部分指标要求至少存在 2 个簇，并且簇数量不能等于样本数量。
    """
    labels = np.asarray(labels)

    unique_labels = set(labels.tolist())
    n_clusters = len(unique_labels - {-1})
    n_noise = int(np.sum(labels == -1))

    metrics: dict[str, float | int | None] = {
        "n_clusters": int(n_clusters),
        "n_noise": n_noise,
        "silhouette": None,
        "calinski_harabasz": None,
        "davies_bouldin": None,
    }

    valid_mask = labels != -1
    valid_labels = labels[valid_mask]
    valid_X = X[valid_mask]

    unique_valid_labels = np.unique(valid_labels)

    if len(valid_X) > 2 and 2 <= len(unique_valid_labels) < len(valid_X):
        metrics["silhouette"] = float(silhouette_score(valid_X, valid_labels))
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(valid_X, valid_labels))
        metrics["davies_bouldin"] = float(davies_bouldin_score(valid_X, valid_labels))

    return metrics


def make_2d_embedding(
    X: np.ndarray,
    random_state: int = 42,
) -> np.ndarray:
    """
    使用 PCA 生成二维坐标，用于聚类结果可视化。
    """
    if X.shape[1] == 1:
        return np.column_stack([X[:, 0], np.zeros(X.shape[0])])

    pca = PCA(n_components=2, random_state=random_state)
    return pca.fit_transform(X)


def run_kmeans(
    X: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42,
) -> ClusteringResult:
    """
    运行 KMeans 聚类。
    """
    model = KMeans(
        n_clusters=n_clusters,
        n_init=10,
        random_state=random_state,
    )
    labels = model.fit_predict(X)

    return ClusteringResult(
        algorithm="kmeans",
        labels=labels,
        metrics=calculate_clustering_metrics(X, labels),
        model=model,
        embedding_2d=make_2d_embedding(X, random_state=random_state),
    )


def run_dbscan(
    X: np.ndarray,
    eps: float = 1.5,
    min_samples: int = 8,
    random_state: int = 42,
) -> ClusteringResult:
    """
    运行 DBSCAN 聚类。

    DBSCAN 不需要预先指定簇数量，但对 eps 较敏感。
    """
    _ = random_state

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
    )
    labels = model.fit_predict(X)

    return ClusteringResult(
        algorithm="dbscan",
        labels=labels,
        metrics=calculate_clustering_metrics(X, labels),
        model=model,
        embedding_2d=make_2d_embedding(X, random_state=42),
    )


def run_agglomerative(
    X: np.ndarray,
    n_clusters: int = 3,
    linkage: str = "ward",
    random_state: int = 42,
) -> ClusteringResult:
    """
    运行层次聚类。
    """
    _ = random_state

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
    )
    labels = model.fit_predict(X)

    return ClusteringResult(
        algorithm="agglomerative",
        labels=labels,
        metrics=calculate_clustering_metrics(X, labels),
        model=model,
        embedding_2d=make_2d_embedding(X, random_state=42),
    )


def run_all_clustering(
    X: pd.DataFrame,
    n_clusters: int = 3,
    dbscan_eps: float = 1.5,
    dbscan_min_samples: int = 8,
    scaler: str = "standard",
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, ClusteringResult], np.ndarray]:
    """
    运行全部聚类算法。
    """
    X_processed, _ = prepare_clustering_matrix(X, scaler=scaler)

    results = {
        "kmeans": run_kmeans(
            X_processed,
            n_clusters=n_clusters,
            random_state=random_state,
        ),
        "dbscan": run_dbscan(
            X_processed,
            eps=dbscan_eps,
            min_samples=dbscan_min_samples,
            random_state=random_state,
        ),
        "agglomerative": run_agglomerative(
            X_processed,
            n_clusters=n_clusters,
            random_state=random_state,
        ),
    }

    rows = []
    for name, result in results.items():
        row = {"algorithm": name}
        row.update(result.metrics)
        rows.append(row)

    summary = pd.DataFrame(rows)

    return summary, results, X_processed


def make_cluster_assignment_table(
    results: dict[str, ClusteringResult],
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """
    生成每个样本的聚类标签表。
    """
    if not results:
        raise ValueError("聚类结果为空。")

    first_result = next(iter(results.values()))
    n_samples = len(first_result.labels)

    if index is None:
        index = pd.RangeIndex(n_samples)

    table = pd.DataFrame(index=index)

    for name, result in results.items():
        table[f"{name}_label"] = result.labels

    return table.reset_index(drop=True)


def make_cluster_embedding_table(
    result: ClusteringResult,
) -> pd.DataFrame:
    """
    生成二维可视化坐标表。
    """
    if result.embedding_2d is None:
        raise ValueError("当前聚类结果没有二维坐标。")

    return pd.DataFrame(
        {
            "x": result.embedding_2d[:, 0],
            "y": result.embedding_2d[:, 1],
            "cluster": result.labels,
        }
    )