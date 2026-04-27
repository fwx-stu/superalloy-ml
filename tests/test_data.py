from pathlib import Path

import numpy as np
import pandas as pd

from superalloy_ml.constants import ELEMENTS, TARGET_COLUMNS
from superalloy_ml.data import (
    generate_synthetic_superalloy_dataset,
    read_table,
    save_processed_dataset,
    split_features_targets,
)
from superalloy_ml.validation import check_dataset_ready


def test_generate_synthetic_dataset_shape_and_columns():
    df = generate_synthetic_superalloy_dataset(n_samples=100, random_state=42)

    assert df.shape[0] == 100

    for element in ELEMENTS:
        assert element in df.columns

    for target in TARGET_COLUMNS:
        assert target in df.columns


def test_composition_sums_to_100():
    df = generate_synthetic_superalloy_dataset(n_samples=50, random_state=42)
    sums = df[ELEMENTS].sum(axis=1).to_numpy()

    assert np.allclose(sums, 100.0)


def test_dataset_validation_passes():
    df = generate_synthetic_superalloy_dataset(n_samples=50, random_state=42)
    check_dataset_ready(df, target="creep_log_life_h")


def test_split_features_targets():
    df = generate_synthetic_superalloy_dataset(n_samples=50, random_state=42)

    X, y = split_features_targets(df, target="creep_log_life_h")

    assert "creep_log_life_h" not in X.columns
    assert len(X) == len(y)
    assert len(y) == 50


def test_read_csv_table(tmp_path: Path):
    path = tmp_path / "sample.csv"
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df.to_csv(path, index=False)

    loaded = read_table(path)

    assert loaded.shape == (2, 2)


def test_read_excel_table(tmp_path: Path):
    path = tmp_path / "sample.xlsx"
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df.to_excel(path, index=False)

    loaded = read_table(path)

    assert loaded.shape == (2, 2)


def test_save_processed_dataset(tmp_path: Path):
    path = tmp_path / "processed.csv"
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    save_processed_dataset(df, path)

    assert path.exists()