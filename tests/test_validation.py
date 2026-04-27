import pandas as pd
import pytest

from superalloy_ml.data import generate_synthetic_superalloy_dataset
from superalloy_ml.validation import (
    check_composition_validity,
    check_dataset_ready,
    check_no_missing_values,
    check_required_columns,
    summarize_dataset,
)


def test_check_required_columns_passes():
    df = pd.DataFrame({"a": [1], "b": [2]})
    check_required_columns(df, ["a", "b"])


def test_check_required_columns_fails():
    df = pd.DataFrame({"a": [1]})

    with pytest.raises(ValueError):
        check_required_columns(df, ["a", "b"])


def test_check_no_missing_values_fails():
    df = pd.DataFrame({"a": [1, None]})

    with pytest.raises(ValueError):
        check_no_missing_values(df)


def test_check_composition_validity_passes():
    df = generate_synthetic_superalloy_dataset(n_samples=20, random_state=42)
    check_composition_validity(df)


def test_check_dataset_ready_passes():
    df = generate_synthetic_superalloy_dataset(n_samples=20, random_state=42)
    check_dataset_ready(df, target="creep_log_life_h")


def test_summarize_dataset():
    df = generate_synthetic_superalloy_dataset(n_samples=20, random_state=42)
    summary = summarize_dataset(df)

    assert summary["n_rows"] == 20
    assert summary["n_columns"] == df.shape[1]
    assert summary["missing_values"] == 0