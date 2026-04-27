from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from superalloy_ml.utils import ensure_dir


def _to_dense_array(X) -> np.ndarray:
    """
    将稀疏矩阵或普通矩阵统一转换为 numpy 数组。
    """
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)


def get_pipeline_model(model: Any) -> Any:
    """
    从 sklearn Pipeline 中取出最终模型。
    """
    if isinstance(model, Pipeline):
        return model.named_steps.get("model")

    return model


def get_pipeline_preprocessor(model: Any) -> Any | None:
    """
    从 sklearn Pipeline 中取出预处理器。
    """
    if isinstance(model, Pipeline):
        return model.named_steps.get("preprocessor")

    return None


def get_feature_names(
    model: Any,
    X: pd.DataFrame,
) -> list[str]:
    """
    获取模型输入特征名。

    如果模型是 Pipeline，则优先从预处理器中获取转换后的特征名。
    """
    preprocessor = get_pipeline_preprocessor(model)

    if preprocessor is not None:
        try:
            return list(preprocessor.get_feature_names_out())
        except Exception:
            pass

    return list(X.columns)


def transform_features_for_model(
    model: Any,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    使用 Pipeline 中的预处理器转换特征。

    如果模型不是 Pipeline，则直接返回原始数值矩阵。
    """
    preprocessor = get_pipeline_preprocessor(model)

    if preprocessor is not None:
        return _to_dense_array(preprocessor.transform(X))

    return _to_dense_array(X)


def get_builtin_feature_importance(
    model: Any,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    提取模型自带的特征重要性。

    支持：
    - 树模型的 feature_importances_
    - 线性模型的 coef_
    """
    estimator = get_pipeline_model(model)
    feature_names = get_feature_names(model, X)

    if estimator is None:
        return pd.DataFrame(columns=["feature", "importance"])

    if hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_, dtype=float)

    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)

        if coef.ndim == 1:
            importance = np.abs(coef)
        else:
            importance = np.mean(np.abs(coef), axis=0)

    else:
        return pd.DataFrame(columns=["feature", "importance"])

    n = min(len(feature_names), len(importance))

    table = pd.DataFrame(
        {
            "feature": feature_names[:n],
            "importance": importance[:n],
        }
    )

    return table.sort_values("importance", ascending=False).reset_index(drop=True)


def compute_permutation_importance(
    model: Any,
    X: pd.DataFrame,
    y,
    scoring: str,
    n_repeats: int = 8,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    计算 Permutation Importance。

    该方法会重复打乱每个特征，观察模型性能下降幅度。
    """
    result = permutation_importance(
        model,
        X,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )

    table = pd.DataFrame(
        {
            "feature": list(X.columns),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )

    return table.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def compute_shap_summary(
    model: Any,
    X: pd.DataFrame,
    max_samples: int = 200,
) -> pd.DataFrame:
    """
    计算 SHAP 平均绝对贡献表。

    如果未安装 shap，会抛出 ImportError。
    """
    import shap

    estimator = get_pipeline_model(model)
    feature_names = get_feature_names(model, X)

    if estimator is None:
        raise ValueError("无法从模型中获取最终估计器。")

    X_sample = X.head(max_samples).copy()
    X_transformed = transform_features_for_model(model, X_sample)

    explainer = shap.Explainer(estimator, X_transformed, feature_names=feature_names)
    explanation = explainer(X_transformed)

    values = np.asarray(explanation.values)

    if values.ndim == 3:
        mean_abs = np.mean(np.abs(values), axis=(0, 2))
    elif values.ndim == 2:
        mean_abs = np.mean(np.abs(values), axis=0)
    elif values.ndim == 1:
        mean_abs = np.abs(values)
    else:
        raise ValueError(f"无法处理的 SHAP values 维度: {values.shape}")

    n = min(len(feature_names), len(mean_abs))

    table = pd.DataFrame(
        {
            "feature": feature_names[:n],
            "mean_abs_shap": mean_abs[:n],
        }
    )

    return table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def plot_importance_bar(
    table: pd.DataFrame,
    value_column: str,
    path: str | Path,
    title: str,
    feature_column: str = "feature",
    top_k: int = 20,
) -> None:
    """
    绘制特征重要性条形图。
    """
    path = Path(path)
    ensure_dir(path.parent)

    if table.empty:
        raise ValueError("特征重要性表为空，无法绘图。")

    if feature_column not in table.columns:
        raise ValueError(f"缺少特征列: {feature_column}")

    if value_column not in table.columns:
        raise ValueError(f"缺少数值列: {value_column}")

    plot_df = table.sort_values(value_column, ascending=False).head(top_k)
    plot_df = plot_df.iloc[::-1]

    plt.figure(figsize=(9, 7))
    plt.barh(plot_df[feature_column], plot_df[value_column])
    plt.xlabel(value_column)
    plt.ylabel("特征")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_explainability_tables(
    output_dir: str | Path,
    builtin_table: pd.DataFrame | None = None,
    permutation_table: pd.DataFrame | None = None,
    shap_table: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """
    保存解释性分析结果表。
    """
    output_dir = ensure_dir(output_dir)
    saved_paths: dict[str, Path] = {}

    if builtin_table is not None and not builtin_table.empty:
        path = output_dir / "builtin_feature_importance.csv"
        builtin_table.to_csv(path, index=False)
        saved_paths["builtin"] = path

    if permutation_table is not None and not permutation_table.empty:
        path = output_dir / "permutation_importance.csv"
        permutation_table.to_csv(path, index=False)
        saved_paths["permutation"] = path

    if shap_table is not None and not shap_table.empty:
        path = output_dir / "shap_importance.csv"
        shap_table.to_csv(path, index=False)
        saved_paths["shap"] = path

    return saved_paths