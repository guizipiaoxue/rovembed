from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class UWCNN(nn.Module):
    """PyTorch implementation of the original TensorFlow UWCNN test model.

    The network predicts a residual image and adds it to the normalized input.
    Inputs and outputs are expected to be NCHW tensors in the [-1, 1] range.
    """

    def __init__(self, in_channels: int = 3, features: int = 16) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, features, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)

        concat1_channels = in_channels + features * 3
        self.conv4 = nn.Conv2d(concat1_channels, features, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)

        concat2_channels = concat1_channels + features * 3
        self.conv7 = nn.Conv2d(concat2_channels, features, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)

        concat3_channels = concat2_channels + features * 3
        self.conv10 = nn.Conv2d(concat3_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.reset_parameters()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        conv1 = F.relu(self.conv1(image), inplace=False)
        conv2 = F.relu(self.conv2(conv1), inplace=False)
        conv3 = F.relu(self.conv3(conv2), inplace=False)
        concat1 = torch.cat([conv1, conv2, conv3, image], dim=1)

        conv4 = F.relu(self.conv4(concat1), inplace=False)
        conv5 = F.relu(self.conv5(conv4), inplace=False)
        conv6 = F.relu(self.conv6(conv5), inplace=False)
        concat2 = torch.cat([concat1, conv4, conv5, conv6], dim=1)

        conv7 = F.relu(self.conv7(concat2), inplace=False)
        conv8 = F.relu(self.conv8(conv7), inplace=False)
        conv9 = F.relu(self.conv9(conv8), inplace=False)
        concat3 = torch.cat([concat2, conv7, conv8, conv9], dim=1)

        residual = self.conv10(concat3)
        return image + residual

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
                nn.init.zeros_(module.bias)
