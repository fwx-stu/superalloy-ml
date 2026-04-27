from __future__ import annotations

import numpy as np
import pandas as pd

from superalloy_ml.constants import (
    APPROX_COST_INDEX,
    ATOMIC_RADIUS_PM,
    ATOMIC_WEIGHT,
    DENSITY_G_CM3,
    ELECTRONEGATIVITY,
    ELEMENTS,
    GAMMA_PRIME_FORMERS,
    OXIDATION_RESISTANCE_ELEMENTS,
    REFRACTORY_ELEMENTS,
)


def validate_composition_columns(df: pd.DataFrame, elements: list[str] | None = None) -> None:
    elements = elements or ELEMENTS
    missing = [element for element in elements if element not in df.columns]

    if missing:
        raise ValueError(f"缺少元素成分列: {missing}")


def composition_sum(df: pd.DataFrame, elements: list[str] | None = None) -> pd.Series:
    elements = elements or ELEMENTS
    validate_composition_columns(df, elements)
    return df[elements].sum(axis=1)


def normalize_composition(df: pd.DataFrame, elements: list[str] | None = None) -> pd.DataFrame:
    """
    将元素成分归一化，使每一行元素总和为 100。

    默认元素成分单位为百分比。
    """
    elements = elements or ELEMENTS
    validate_composition_columns(df, elements)

    out = df.copy()
    total = out[elements].sum(axis=1).replace(0, np.nan)

    out[elements] = out[elements].div(total, axis=0) * 100.0
    out[elements] = out[elements].fillna(0.0)

    return out


def weighted_property(
    df: pd.DataFrame,
    property_map: dict[str, float],
    elements: list[str] | None = None,
) -> pd.Series:
    """
    计算元素物性的成分加权平均值。
    """
    elements = elements or ELEMENTS
    validate_composition_columns(df, elements)

    weights = df[elements].to_numpy(dtype=float) / 100.0
    values = np.array([property_map[element] for element in elements], dtype=float)

    return pd.Series(weights @ values, index=df.index)


def weighted_std_property(
    df: pd.DataFrame,
    property_map: dict[str, float],
    elements: list[str] | None = None,
) -> pd.Series:
    """
    计算元素物性的加权标准差，用于表示物性失配。
    """
    elements = elements or ELEMENTS
    validate_composition_columns(df, elements)

    weights = df[elements].to_numpy(dtype=float) / 100.0
    values = np.array([property_map[element] for element in elements], dtype=float)

    mean = weights @ values
    diff_squared = (values[None, :] - mean[:, None]) ** 2
    variance = np.sum(weights * diff_squared, axis=1)

    return pd.Series(np.sqrt(variance), index=df.index)


def calculate_mixing_entropy(df: pd.DataFrame, elements: list[str] | None = None) -> pd.Series:
    """
    计算理想混合熵的无量纲形式：

        S_mix / R = -sum(c_i * ln(c_i))
    """
    elements = elements or ELEMENTS
    validate_composition_columns(df, elements)

    c = df[elements].to_numpy(dtype=float) / 100.0
    c = np.clip(c, 1e-12, 1.0)

    entropy = -np.sum(c * np.log(c), axis=1)
    return pd.Series(entropy, index=df.index)


def add_basic_composition_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加基础成分特征。
    """
    out = df.copy()

    out["composition_sum"] = composition_sum(out)
    out["gamma_prime_formers_total"] = out[GAMMA_PRIME_FORMERS].sum(axis=1)
    out["refractory_total"] = out[REFRACTORY_ELEMENTS].sum(axis=1)
    out["oxidation_resistance_total"] = out[OXIDATION_RESISTANCE_ELEMENTS].sum(axis=1)

    out["al_ti_ratio"] = out["Al"] / (out["Ti"] + 1e-8)
    out["ta_w_ratio"] = out["Ta"] / (out["W"] + 1e-8)
    out["cr_al_ratio"] = out["Cr"] / (out["Al"] + 1e-8)
    out["co_ni_ratio"] = out["Co"] / (out["Ni"] + 1e-8)

    return out


def add_physical_descriptor_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加基于元素物性的描述符。
    """
    out = df.copy()

    out["avg_atomic_weight"] = weighted_property(out, ATOMIC_WEIGHT)
    out["avg_atomic_radius_pm"] = weighted_property(out, ATOMIC_RADIUS_PM)
    out["atomic_radius_mismatch"] = weighted_std_property(out, ATOMIC_RADIUS_PM)

    out["avg_electronegativity"] = weighted_property(out, ELECTRONEGATIVITY)
    out["electronegativity_mismatch"] = weighted_std_property(out, ELECTRONEGATIVITY)

    out["estimated_density_g_cm3"] = weighted_property(out, DENSITY_G_CM3)
    out["estimated_cost_index"] = weighted_property(out, APPROX_COST_INDEX)
    out["mixing_entropy_r"] = calculate_mixing_entropy(out)

    return out


def add_processing_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加热处理和测试条件相关特征。
    """
    required_columns = [
        "solution_temp_c",
        "solution_time_h",
        "aging_temp_c",
        "aging_time_h",
        "cooling_rate_c_s",
        "test_temp_c",
        "stress_mpa",
    ]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"缺少工艺或测试条件列: {missing}")

    out = df.copy()

    out["solution_temp_k"] = out["solution_temp_c"] + 273.15
    out["aging_temp_k"] = out["aging_temp_c"] + 273.15
    out["test_temp_k"] = out["test_temp_c"] + 273.15

    out["solution_time_log"] = np.log1p(out["solution_time_h"])
    out["aging_time_log"] = np.log1p(out["aging_time_h"])
    out["cooling_rate_log"] = np.log1p(out["cooling_rate_c_s"])

    out["solution_parameter"] = out["solution_temp_k"] * out["solution_time_log"]
    out["aging_parameter"] = out["aging_temp_k"] * out["aging_time_log"]

    out["temp_gap_solution_aging"] = out["solution_temp_c"] - out["aging_temp_c"]
    out["test_temp_fraction_of_solution"] = out["test_temp_k"] / (out["solution_temp_k"] + 1e-8)

    return out


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加交互特征。
    """
    out = df.copy()

    if "gamma_prime_formers_total" not in out.columns:
        out = add_basic_composition_features(out)

    if "test_temp_k" not in out.columns:
        out = add_processing_features(out)

    out["stress_temp_interaction"] = out["stress_mpa"] * out["test_temp_k"]
    out["refractory_temp_interaction"] = out["refractory_total"] / (out["test_temp_k"] + 1e-8)
    out["gamma_prime_aging_interaction"] = out["gamma_prime_formers_total"] * out["aging_temp_k"]
    out["cr_oxidation_temp_interaction"] = out["Cr"] / (out["test_temp_k"] + 1e-8)

    return out


def make_features(
    df: pd.DataFrame,
    normalize: bool = True,
    add_composition: bool = True,
    add_physical: bool = True,
    add_processing: bool = True,
    add_interactions: bool = True,
) -> pd.DataFrame:
    """
    特征工程统一入口。
    """
    out = df.copy()

    if normalize:
        out = normalize_composition(out)

    if add_composition:
        out = add_basic_composition_features(out)

    if add_physical:
        out = add_physical_descriptor_features(out)

    if add_processing:
        out = add_processing_features(out)

    if add_interactions:
        out = add_interaction_features(out)

    return out


def get_numeric_feature_columns(df: pd.DataFrame, target_columns: list[str]) -> list[str]:
    """
    获取可用于建模的数值特征列。
    """
    excluded = set(target_columns)
    feature_columns = []

    for column in df.columns:
        if column in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            feature_columns.append(column)

    return feature_columns