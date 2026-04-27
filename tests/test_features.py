import numpy as np
import pandas as pd

from superalloy_ml.constants import ELEMENTS
from superalloy_ml.features import (
    calculate_mixing_entropy,
    make_features,
    normalize_composition,
    weighted_property,
)


def _small_composition_df():
    return pd.DataFrame(
        {
            "Ni": [60.0, 70.0],
            "Co": [10.0, 5.0],
            "Cr": [10.0, 10.0],
            "Al": [5.0, 4.0],
            "Ti": [3.0, 2.0],
            "Ta": [3.0, 2.0],
            "W": [3.0, 2.0],
            "Mo": [2.0, 2.0],
            "Re": [1.0, 1.0],
            "Nb": [1.0, 1.0],
            "Hf": [1.0, 1.0],
            "C": [0.1, 0.1],
            "B": [0.1, 0.1],
            "solution_temp_c": [1150.0, 1180.0],
            "solution_time_h": [4.0, 5.0],
            "aging_temp_c": [800.0, 820.0],
            "aging_time_h": [16.0, 20.0],
            "cooling_rate_c_s": [10.0, 15.0],
            "test_temp_c": [850.0, 900.0],
            "stress_mpa": [300.0, 350.0],
        }
    )


def test_normalize_composition():
    df = _small_composition_df()
    out = normalize_composition(df)

    sums = out[ELEMENTS].sum(axis=1)

    assert np.allclose(sums, 100.0)


def test_weighted_property_returns_values():
    df = normalize_composition(_small_composition_df())

    property_map = {element: index + 1 for index, element in enumerate(ELEMENTS)}
    values = weighted_property(df, property_map)

    assert len(values) == len(df)
    assert np.all(np.isfinite(values))


def test_mixing_entropy_positive():
    df = normalize_composition(_small_composition_df())
    entropy = calculate_mixing_entropy(df)

    assert len(entropy) == len(df)
    assert np.all(entropy > 0)


def test_make_features_adds_expected_columns():
    df = _small_composition_df()
    out = make_features(df)

    expected_columns = [
        "composition_sum",
        "gamma_prime_formers_total",
        "refractory_total",
        "oxidation_resistance_total",
        "avg_atomic_weight",
        "avg_atomic_radius_pm",
        "atomic_radius_mismatch",
        "avg_electronegativity",
        "estimated_density_g_cm3",
        "estimated_cost_index",
        "mixing_entropy_r",
        "test_temp_k",
        "stress_temp_interaction",
    ]

    for column in expected_columns:
        assert column in out.columns