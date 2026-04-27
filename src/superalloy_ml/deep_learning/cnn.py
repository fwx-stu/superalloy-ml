from __future__ import annotations

import torch
from torch import nn


class SimpleCNNClassifier(nn.Module):
    """
    简单 CNN 分类模型。

    输入形状：
    [batch, channels, height, width]
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 3,
        base_channels: int = 16,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(base_channels),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(base_channels * 2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(base_channels * 4, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)