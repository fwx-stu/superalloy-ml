from __future__ import annotations

import torch
from torch import nn


class RNNClassifier(nn.Module):
    """
    RNN 序列分类模型。

    输入形状：
    [batch, sequence_length, n_features]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_classes: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, hidden = self.rnn(x)
        _ = output
        last_hidden = hidden[-1]
        return self.classifier(last_hidden)


class LSTMClassifier(nn.Module):
    """
    LSTM 序列分类模型。

    输入形状：
    [batch, sequence_length, n_features]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_classes: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, (hidden, cell) = self.lstm(x)
        _ = output, cell
        last_hidden = hidden[-1]
        return self.classifier(last_hidden)