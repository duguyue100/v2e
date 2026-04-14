from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Down(nn.Module):  # type: ignore[misc]
    def __init__(self, inChannels: int, outChannels: int, filterSize: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            inChannels,
            outChannels,
            filterSize,
            stride=1,
            padding=int((filterSize - 1) / 2),
        )
        self.conv2 = nn.Conv2d(
            outChannels,
            outChannels,
            filterSize,
            stride=1,
            padding=int((filterSize - 1) / 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.avg_pool2d(x, 2)
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        return x


class Up(nn.Module):  # type: ignore[misc]
    def __init__(self, inChannels: int, outChannels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inChannels, outChannels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(2 * outChannels, outChannels, 3, stride=1, padding=1)

    def forward(self, x: torch.Tensor, skpCn: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(torch.cat((x, skpCn), 1)), negative_slope=0.1)
        return x


class UNet(nn.Module):  # type: ignore[misc]
    def __init__(self, inChannels: int, outChannels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inChannels, 32, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.down1 = Down(32, 64, 5)
        self.down2 = Down(64, 128, 3)
        self.down3 = Down(128, 256, 3)
        self.down4 = Down(256, 512, 3)
        self.down5 = Down(512, 512, 3)
        self.up1 = Up(512, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.up5 = Up(64, 32)
        self.conv3 = nn.Conv2d(32, outChannels, 3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        s1 = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)
        x = self.down5(s5)
        x = self.up1(x, s5)
        x = self.up2(x, s4)
        x = self.up3(x, s3)
        x = self.up4(x, s2)
        x = self.up5(x, s1)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.1)
        return x


class BackWarp(nn.Module):  # type: ignore[misc]
    def __init__(self, W: int, H: int, device: Any) -> None:
        super().__init__()
        gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
        self.W = W
        self.H = H
        self.gridX = torch.tensor(gridX, requires_grad=False, device=device)
        self.gridY = torch.tensor(gridY, requires_grad=False, device=device)

    def forward(self, img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        x = self.gridX.unsqueeze(0).expand_as(u).float() + u
        y = self.gridY.unsqueeze(0).expand_as(v).float() + v
        x = 2 * (x / self.W - 0.5)
        y = 2 * (y / self.H - 0.5)
        grid = torch.stack((x, y), dim=3)
        imgOut = torch.nn.functional.grid_sample(img, grid)
        return imgOut


t = np.linspace(0.125, 0.875, 7)
