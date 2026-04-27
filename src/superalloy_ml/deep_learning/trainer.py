from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from superalloy_ml.evaluation import classification_metrics, regression_metrics


@dataclass
class TrainingHistory:
    train_loss: list[float]
    val_loss: list[float]


def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_torch_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int = 10,
    device: torch.device | None = None,
    task: str = "classification",
    verbose: bool = True,
) -> TrainingHistory:
    """
    通用训练循环。
    """
    device = device or get_device()
    model.to(device)

    history = TrainingHistory(train_loss=[], val_loss=[])

    for epoch in range(epochs):
        model.train()
        batch_losses = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            output = model(X_batch)

            if task == "regression":
                loss = loss_fn(output, y_batch.float())
            else:
                loss = loss_fn(output, y_batch.long())

            loss.backward()
            optimizer.step()

            batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses))
        history.train_loss.append(train_loss)

        if val_loader is not None:
            val_loss = evaluate_loss(
                model=model,
                data_loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                task=task,
            )
        else:
            val_loss = float("nan")

        history.val_loss.append(val_loss)

        if verbose:
            print(
                f"epoch={epoch + 1:03d}, "
                f"train_loss={train_loss:.6f}, "
                f"val_loss={val_loss:.6f}"
            )

    return history


def evaluate_loss(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    task: str,
) -> float:
    model.eval()
    losses = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            output = model(X_batch)

            if task == "regression":
                loss = loss_fn(output, y_batch.float())
            else:
                loss = loss_fn(output, y_batch.long())

            losses.append(float(loss.item()))

    return float(np.mean(losses))


def predict_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device | None = None,
    task: str = "classification",
) -> tuple[np.ndarray, np.ndarray]:
    """
    返回真实值和预测值。
    """
    device = device or get_device()
    model.to(device)
    model.eval()

    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            output = model(X_batch)

            if task == "regression":
                pred = output.detach().cpu().numpy().reshape(-1)
                true = y_batch.detach().cpu().numpy().reshape(-1)
            else:
                pred = torch.argmax(output, dim=1).detach().cpu().numpy()
                true = y_batch.detach().cpu().numpy()

            y_true_list.append(true)
            y_pred_list.append(pred)

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)

    return y_true, y_pred


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device | None = None,
    task: str = "classification",
) -> dict[str, float]:
    """
    计算模型评估指标。
    """
    y_true, y_pred = predict_model(
        model=model,
        data_loader=data_loader,
        device=device,
        task=task,
    )

    if task == "regression":
        return regression_metrics(y_true, y_pred)

    return classification_metrics(y_true, y_pred)


def save_torch_model(model: nn.Module, path) -> None:
    """
    保存 PyTorch 模型权重。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_torch_weights(model: nn.Module, path, device: torch.device | None = None) -> nn.Module:
    """
    加载 PyTorch 模型权重。
    """
    device = device or get_device(prefer_gpu=False)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model