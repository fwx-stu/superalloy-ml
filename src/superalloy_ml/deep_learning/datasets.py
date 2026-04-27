from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from superalloy_ml.data import split_features_targets
from superalloy_ml.preprocessing import create_preprocessor

TaskType = Literal["regression", "classification"]


@dataclass
class TabularDataBundle:
    X_train: torch.Tensor
    X_test: torch.Tensor
    y_train: torch.Tensor
    y_test: torch.Tensor
    input_dim: int
    output_dim: int
    preprocessor: object


def _to_dense_array(X) -> np.ndarray:
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)


def prepare_tabular_tensors(
    df: pd.DataFrame,
    target: str,
    task: TaskType,
    test_size: float = 0.2,
    random_state: int = 42,
    scaler: str = "standard",
) -> TabularDataBundle:
    """
    将表格数据转换为 PyTorch Tensor。

    回归任务：
    - y 输出形状为 [N, 1]

    分类任务：
    - y 输出形状为 [N]
    - 标签类型为 long
    """
    X, y = split_features_targets(df, target=target)

    stratify = y if task == "classification" else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    preprocessor = create_preprocessor(X_train, scaler=scaler)

    X_train_np = _to_dense_array(preprocessor.fit_transform(X_train)).astype(np.float32)
    X_test_np = _to_dense_array(preprocessor.transform(X_test)).astype(np.float32)

    if task == "regression":
        y_train_np = y_train.to_numpy(dtype=np.float32).reshape(-1, 1)
        y_test_np = y_test.to_numpy(dtype=np.float32).reshape(-1, 1)
        output_dim = 1

        y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32)

    elif task == "classification":
        y_train_np = y_train.to_numpy(dtype=np.int64)
        y_test_np = y_test.to_numpy(dtype=np.int64)

        classes = np.unique(y.to_numpy(dtype=np.int64))
        output_dim = int(classes.max()) + 1

        y_train_tensor = torch.tensor(y_train_np, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test_np, dtype=torch.long)

    else:
        raise ValueError(f"未知任务类型: {task}")

    return TabularDataBundle(
        X_train=torch.tensor(X_train_np, dtype=torch.float32),
        X_test=torch.tensor(X_test_np, dtype=torch.float32),
        y_train=y_train_tensor,
        y_test=y_test_tensor,
        input_dim=int(X_train_np.shape[1]),
        output_dim=int(output_dim),
        preprocessor=preprocessor,
    )


def make_loaders_from_tensors(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader]:
    """
    从 Tensor 构建 DataLoader。
    """
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader


def make_synthetic_image_dataset(
    n_samples: int = 300,
    image_size: int = 16,
    n_classes: int = 3,
    random_state: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    生成简单二维图像数据，用于 CNN 示例。

    类别 0：竖向亮带
    类别 1：横向亮带
    类别 2：对角亮带
    """
    rng = np.random.default_rng(random_state)

    images = rng.normal(
        loc=0.0,
        scale=0.25,
        size=(n_samples, 1, image_size, image_size),
    ).astype(np.float32)

    labels = np.arange(n_samples) % n_classes
    rng.shuffle(labels)

    for i, label in enumerate(labels):
        if label == 0:
            col = image_size // 2
            images[i, 0, :, col - 1 : col + 1] += 1.5
        elif label == 1:
            row = image_size // 2
            images[i, 0, row - 1 : row + 1, :] += 1.5
        else:
            for j in range(image_size):
                images[i, 0, j, j] += 1.5

    return torch.tensor(images, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def make_synthetic_sequence_dataset(
    n_samples: int = 300,
    sequence_length: int = 20,
    n_features: int = 6,
    n_classes: int = 3,
    random_state: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    生成简单序列数据，用于 RNN、LSTM、Transformer 示例。

    每个类别对应不同的趋势模式。
    """
    rng = np.random.default_rng(random_state)

    X = rng.normal(
        loc=0.0,
        scale=0.4,
        size=(n_samples, sequence_length, n_features),
    ).astype(np.float32)

    y = np.arange(n_samples) % n_classes
    rng.shuffle(y)

    t = np.linspace(0, 1, sequence_length, dtype=np.float32)

    for i, label in enumerate(y):
        if label == 0:
            X[i, :, 0] += t * 2.0
        elif label == 1:
            X[i, :, 1] += (1.0 - t) * 2.0
        else:
            X[i, :, 2] += np.sin(2 * np.pi * t) * 1.5

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def split_tensor_dataset(
    X: torch.Tensor,
    y: torch.Tensor,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    对 Tensor 数据进行训练集和测试集划分。
    """
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y.numpy() if y.ndim == 1 else None,
    )

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]