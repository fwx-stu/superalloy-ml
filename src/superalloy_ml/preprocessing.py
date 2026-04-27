from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]


def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]


def create_preprocessor(
    X: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_strategy: str = "most_frequent",
    scaler: str = "standard",
) -> ColumnTransformer:
    """
    创建 sklearn 预处理流水线。

    数值特征：
    - 缺失值填充
    - 标准化或归一化

    类别特征：
    - 缺失值填充
    - One-Hot 编码
    """
    numeric_columns = get_numeric_columns(X)
    categorical_columns = get_categorical_columns(X)

    if scaler == "standard":
        scaler_step = StandardScaler()
    elif scaler == "minmax":
        scaler_step = MinMaxScaler()
    elif scaler == "none":
        scaler_step = "passthrough"
    else:
        raise ValueError(f"未知 scaler: {scaler}")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=numeric_strategy)),
            ("scaler", scaler_step),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=categorical_strategy)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []

    if numeric_columns:
        transformers.append(("num", numeric_pipeline, numeric_columns))

    if categorical_columns:
        transformers.append(("cat", categorical_pipeline, categorical_columns))

    if not transformers:
        raise ValueError("没有可用于预处理的特征列。")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def preprocess_fit_transform(
    X: pd.DataFrame,
    scaler: str = "standard",
):
    """
    拟合并转换特征。
    """
    preprocessor = create_preprocessor(X, scaler=scaler)
    X_processed = preprocessor.fit_transform(X)
    return X_processed, preprocessor


def preprocess_transform(
    X: pd.DataFrame,
    preprocessor: ColumnTransformer,
):
    """
    使用已拟合的预处理器转换特征。
    """
    return preprocessor.transform(X)


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """
    获取预处理后的特征名。
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return []