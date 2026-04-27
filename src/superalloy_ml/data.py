from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from superalloy_ml.constants import ELEMENTS, TARGET_COLUMNS
from superalloy_ml.features import make_features
from superalloy_ml.utils import ensure_dir, resolve_path


def _generate_composition(
    rng: np.random.Generator,
    n_samples: int,
) -> pd.DataFrame:
    """
    生成近似高温合金成分。

    Ni 为基体元素，其余元素用于构造合金化特征。
    """
    alpha = np.array(
        [
            55.0,
            8.0,
            12.0,
            5.0,
            3.0,
            2.0,
            2.0,
            1.5,
            0.4,
            0.8,
            0.3,
            0.1,
            0.1,
        ]
    )

    composition = rng.dirichlet(alpha, size=n_samples) * 100.0
    df = pd.DataFrame(composition, columns=ELEMENTS)

    df["C"] = np.clip(df["C"], 0.01, 0.25)
    df["B"] = np.clip(df["B"], 0.005, 0.15)

    non_ni_elements = [element for element in ELEMENTS if element != "Ni"]
    total_non_ni = df[non_ni_elements].sum(axis=1)
    df["Ni"] = 100.0 - total_non_ni

    df[ELEMENTS] = df[ELEMENTS].clip(lower=0.0)

    total = df[ELEMENTS].sum(axis=1)
    df[ELEMENTS] = df[ELEMENTS].div(total, axis=0) * 100.0

    return df


def _generate_process_and_test_conditions(
    rng: np.random.Generator,
    n_samples: int,
) -> pd.DataFrame:
    """
    生成热处理参数和测试条件。
    """
    return pd.DataFrame(
        {
            "solution_temp_c": rng.uniform(1050, 1250, size=n_samples),
            "solution_time_h": rng.uniform(1, 8, size=n_samples),
            "aging_temp_c": rng.uniform(650, 900, size=n_samples),
            "aging_time_h": rng.uniform(4, 32, size=n_samples),
            "cooling_rate_c_s": rng.lognormal(mean=np.log(10), sigma=0.8, size=n_samples),
            "test_temp_c": rng.uniform(650, 1050, size=n_samples),
            "stress_mpa": rng.uniform(120, 650, size=n_samples),
        }
    )


def _add_targets(
    df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    根据简化规律生成目标值。

    这些公式仅用于构造可运行的数据流程。
    """
    out = df.copy()

    gamma_prime = out["gamma_prime_formers_total"]
    refractory = out["refractory_total"]
    oxidation = out["oxidation_resistance_total"]
    density = out["estimated_density_g_cm3"]
    radius_mismatch = out["atomic_radius_mismatch"]
    cost = out["estimated_cost_index"]

    test_temp = out["test_temp_c"]
    stress = out["stress_mpa"]
    aging_temp = out["aging_temp_c"]
    solution_temp = out["solution_temp_c"]
    aging_time = out["aging_time_h"]
    cooling_rate = out["cooling_rate_c_s"]

    noise_strength = rng.normal(0, 35, size=len(out))
    noise_tensile = rng.normal(0, 40, size=len(out))
    noise_creep = rng.normal(0, 0.25, size=len(out))
    noise_oxidation = rng.normal(0, 0.03, size=len(out))

    yield_strength = (
        420
        + 18 * gamma_prime
        + 12 * refractory
        + 5 * out["Co"]
        + 6 * radius_mismatch
        + 0.08 * (solution_temp - 1100)
        - 0.75 * (test_temp - 700)
        - 0.02 * np.maximum(aging_temp - 800, 0) ** 2
        + 6 * np.log1p(cooling_rate)
        + noise_strength
    )

    tensile_strength = (
        yield_strength
        + 120
        + 4.5 * out["Cr"]
        + 3.0 * out["Co"]
        - 0.25 * (test_temp - 700)
        + noise_tensile
    )

    creep_log_life = (
        7.2
        + 0.10 * refractory
        + 0.07 * gamma_prime
        + 0.04 * out["Re"]
        + 0.015 * out["Ta"]
        - 0.0055 * (test_temp - 700)
        - 0.0038 * (stress - 200)
        + 0.0008 * (solution_temp - 1120)
        + 0.03 * np.log1p(aging_time)
        - 0.04 * density
        - 0.015 * cost
        + noise_creep
    )

    oxidation_mass_gain = (
        0.20
        + 0.0018 * np.maximum(test_temp - 700, 0)
        - 0.012 * oxidation
        + 0.005 * out["Mo"]
        + 0.004 * out["W"]
        + noise_oxidation
    )

    out["yield_strength_mpa"] = np.clip(yield_strength, 50, None)
    out["tensile_strength_mpa"] = np.clip(tensile_strength, 80, None)
    out["creep_log_life_h"] = np.clip(creep_log_life, 0.1, None)
    out["oxidation_mass_gain_mg_cm2"] = np.clip(oxidation_mass_gain, 0.01, None)

    q1, q2 = np.quantile(out["creep_log_life_h"], [0.33, 0.66])
    out["creep_life_class"] = pd.cut(
        out["creep_log_life_h"],
        bins=[-np.inf, q1, q2, np.inf],
        labels=[0, 1, 2],
    ).astype(int)

    return out


def generate_synthetic_superalloy_dataset(
    n_samples: int = 1200,
    random_state: int = 42,
    add_engineered_features: bool = True,
) -> pd.DataFrame:
    """
    生成合成高温合金表格数据。
    """
    rng = np.random.default_rng(random_state)

    composition = _generate_composition(rng, n_samples)
    process = _generate_process_and_test_conditions(rng, n_samples)

    df = pd.concat([composition, process], axis=1)

    if add_engineered_features:
        df = make_features(df, normalize=True)

    df = _add_targets(df, rng)

    return df


def save_synthetic_dataset(
    path: str | Path = "data/processed/synthetic_superalloy.csv",
    n_samples: int = 1200,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    生成并保存合成数据。
    """
    path = resolve_path(path)
    ensure_dir(path.parent)

    df = generate_synthetic_superalloy_dataset(
        n_samples=n_samples,
        random_state=random_state,
        add_engineered_features=True,
    )

    df.to_csv(path, index=False)
    return df


def read_table(path: str | Path) -> pd.DataFrame:
    """
    读取 CSV 或 Excel 表格。
    """
    path = resolve_path(path)

    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError(f"不支持的数据文件格式: {suffix}")


def load_dataset(path: str | Path = "data/processed/synthetic_superalloy.csv") -> pd.DataFrame:
    """
    从指定路径读取数据。
    """
    return read_table(path)


def load_configured_dataset(config: dict) -> pd.DataFrame:
    """
    根据配置读取数据。

    data.source 为 synthetic 时：
    - 如果 synthetic_path 存在，直接读取
    - 如果不存在，自动生成

    data.source 为 external 时：
    - 从 external_path 读取 CSV 或 Excel
    """
    data_config = config.get("data", {})
    project_config = config.get("project", {})

    source = data_config.get("source", "synthetic")
    random_state = int(project_config.get("random_state", 42))

    if source == "synthetic":
        synthetic_path = data_config.get(
            "synthetic_path",
            "data/processed/synthetic_superalloy.csv",
        )
        path = resolve_path(synthetic_path)

        if path.exists():
            return read_table(path)

        return save_synthetic_dataset(
            path=path,
            n_samples=int(data_config.get("n_samples", 1200)),
            random_state=random_state,
        )

    if source == "external":
        external_path = data_config.get("external_path")
        if not external_path:
            raise ValueError("配置中 data.external_path 为空。")
        return read_table(external_path)

    raise ValueError(f"未知数据来源: {source}")


def split_features_targets(
    df: pd.DataFrame,
    target: str,
    drop_all_known_targets: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    拆分特征和目标。
    """
    if target not in df.columns:
        raise ValueError(f"目标列不存在: {target}")

    if drop_all_known_targets:
        drop_columns = [column for column in TARGET_COLUMNS if column in df.columns]
    else:
        drop_columns = [target]

    X = df.drop(columns=drop_columns)
    y = df[target]

    return X, y


def save_processed_dataset(df: pd.DataFrame, path: str | Path) -> None:
    """
    保存处理后的数据。
    """
    path = resolve_path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)