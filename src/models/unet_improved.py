# src/models/unet_improved.py

# [FINAL REPORT CHANGE] Added Dropout2d(0.3) after the bottleneck layer.
# Original U-Net had no regularization, leading to over-segmentation (precision=0.50)
# because the high-capacity bottleneck memorized background noise on the small dataset.
# Dropout at the bottleneck forces the decoder to rely on skip connections more,
# which improves boundary precision without hurting recall significantly.

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNetImproved(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=None):
        super().__init__()

        if features is None:
            features = [64, 128, 256, 512]

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (unchanged)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # [FINAL REPORT CHANGE] Bottleneck now followed by Dropout2d(0.3).
        # Spatial dropout zeros entire feature channels, acting as a strong regularizer
        # and preventing the decoder from over-relying on any single bottleneck channel.
        self.bottleneck = nn.Sequential(
            DoubleConv(features[-1], features[-1] * 2),
            nn.Dropout2d(p=0.3),
        )

        # Decoder (unchanged)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape[2:] != skip_connection.shape[2:]:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=False)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)
