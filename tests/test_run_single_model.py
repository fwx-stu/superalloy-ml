from pathlib import Path

import pandas as pd

from scripts.run_single_model import (
    parse_args,
    resolve_target,
)
from superalloy_ml.data import generate_synthetic_superalloy_dataset
from superalloy_ml.utils import get_config


def test_resolve_regression_target():
    config = {
        "target": {
            "regression": "creep_log_life_h",
            "classification": "creep_life_class",
        }
    }

    class Args:
        target = None
        task = "regression"

    target = resolve_target(Args(), config)

    assert target == "creep_log_life_h"


def test_resolve_classification_target():
    config = {
        "target": {
            "regression": "creep_log_life_h",
            "classification": "creep_life_class",
        }
    }

    class Args:
        target = None
        task = "classification"

    target = resolve_target(Args(), config)

    assert target == "creep_life_class"


def test_external_dataframe_can_be_created(tmp_path: Path):
    df = generate_synthetic_superalloy_dataset(n_samples=20, random_state=42)

    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)

    loaded = pd.read_csv(path)

    assert loaded.shape[0] == 20
    assert "creep_log_life_h" in loaded.columns