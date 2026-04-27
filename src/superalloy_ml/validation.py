from __future__ import annotations

import numpy as np
import pandas as pd

from superalloy_ml.constants import ELEMENTS, TARGET_COLUMNS


def check_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise ValueError(f"缺少必要列: {missing}")


def check_no_missing_values(df: pd.DataFrame) -> None:
    missing = df.isna().sum()
    missing = missing[missing > 0]

    if len(missing) > 0:
        raise ValueError(f"数据中存在缺失值:\n{missing}")


def check_duplicate_rows(df: pd.DataFrame) -> None:
    duplicated = int(df.duplicated().sum())

    if duplicated > 0:
        raise ValueError(f"数据中存在重复行，数量: {duplicated}")


def check_composition_validity(
    df: pd.DataFrame,
    elements: list[str] | None = None,
    atol: float = 1e-5,
    require_sum_100: bool = True,
) -> None:
    """
    检查元素成分是否合法。
    """
    elements = elements or ELEMENTS
    check_required_columns(df, elements)

    composition = df[elements]

    if (composition < 0).any().any():
        raise ValueError("元素成分中存在负值。")

    if require_sum_100:
        sums = composition.sum(axis=1).to_numpy()

        if not np.allclose(sums, 100.0, atol=atol):
            max_error = float(np.max(np.abs(sums - 100.0)))
            raise ValueError(f"元素成分总和不等于 100，最大误差: {max_error}")


def check_target_column(df: pd.DataFrame, target: str) -> None:
    if target not in df.columns:
        raise ValueError(f"目标列不存在: {target}")

    if df[target].isna().any():
        raise ValueError(f"目标列存在缺失值: {target}")


def check_known_target_columns(df: pd.DataFrame) -> None:
    available_targets = [column for column in TARGET_COLUMNS if column in df.columns]

    if not available_targets:
        raise ValueError("未找到已知目标列。")

    for column in available_targets:
        if df[column].isna().any():
            raise ValueError(f"目标列存在缺失值: {column}")


def check_dataset_ready(
    df: pd.DataFrame,
    target: str | None = None,
    check_duplicates: bool = False,
    require_composition_sum_100: bool = True,
) -> None:
    """
    检查数据是否可用于后续流程。
    """
    check_no_missing_values(df)
    check_composition_validity(
        df,
        require_sum_100=require_composition_sum_100,
    )

    if target is None:
        check_known_target_columns(df)
    else:
        check_target_column(df, target)

    if check_duplicates:
        check_duplicate_rows(df)


def summarize_dataset(df: pd.DataFrame) -> dict:
    """
    生成数据摘要。
    """
    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": int(len(df.select_dtypes(include="number").columns)),
        "non_numeric_columns": int(len(df.select_dtypes(exclude="number").columns)),
    }