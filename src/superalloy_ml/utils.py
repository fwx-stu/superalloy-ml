from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: str | os.PathLike) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_random_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_yaml(path: str | os.PathLike) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj: dict[str, Any], path: str | os.PathLike) -> None:
    path = Path(path)
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_json(path: str | os.PathLike) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_section(title: str) -> None:
    line = "=" * 80
    print(line)
    print(title)
    print(line)


def resolve_path(path: str | os.PathLike, root: str | os.PathLike | None = None) -> Path:
    path = Path(path)

    if path.is_absolute():
        return path

    if root is None:
        root = project_root()

    return Path(root) / path


def get_config(path: str | os.PathLike = "configs/default.yaml") -> dict[str, Any]:
    config_path = resolve_path(path)
    return load_yaml(config_path)