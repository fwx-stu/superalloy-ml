from __future__ import annotations

import torch
from torch import nn


class TabularMLP(nn.Module):
    """
    表格数据 MLP。

    可用于：
    - 回归
    - 分类
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        hidden_dims = hidden_dims or [128, 64]

        layers = []
        dims = [input_dim] + hidden_dims

        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(out_dim),
                    nn.Dropout(dropout),
                ]
            )

        layers.append(nn.Linear(dims[-1], output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)