# src/models/cnn_improved.py

# [FINAL REPORT CHANGE] Replaced original CNNAutoencoder with CNNWithSkips.
# Original had no skip connections and no batch normalization, causing it to lose
# all spatial detail through aggressive downsampling and to underfit in only 10 epochs.
# Changes:
#   - Added 3 skip connections (encoder→decoder) to preserve spatial detail
#   - Added BatchNorm2d after every conv for stable gradient flow
#   - Decoder now fuses encoder feature maps via concatenation before refining

import torch
import torch.nn as nn


def _conv_bn_relu(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class CNNWithSkips(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (same channel sizes as original, now with BatchNorm)
        self.enc1 = _conv_bn_relu(3, 16)    # [B, 16, 256, 256]
        self.enc2 = _conv_bn_relu(16, 32)   # [B, 32, 128, 128]
        self.enc3 = _conv_bn_relu(32, 64)   # [B, 64,  64,  64]
        # After pool(enc3): [B, 64, 32, 32] — bottleneck

        # [FINAL REPORT CHANGE] Decoder uses ConvTranspose2d + skip concat + refine conv.
        # Each decoder stage: upsample → concatenate encoder skip → conv to clean channels.
        self.dec3_up   = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)   # →[B,32,64,64]
        self.dec3_conv = _conv_bn_relu(32 + 64, 32)  # cat with enc3 (64ch) → 96ch → 32ch

        self.dec2_up   = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)   # →[B,16,128,128]
        self.dec2_conv = _conv_bn_relu(16 + 32, 16)  # cat with enc2 (32ch) → 48ch → 16ch

        self.dec1_up   = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)    # →[B,8,256,256]
        self.dec1_conv = _conv_bn_relu(8 + 16, 8)   # cat with enc1 (16ch) → 24ch → 8ch

        self.final = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)                   # [B, 16, 256, 256]
        e2 = self.enc2(self.pool(e1))        # [B, 32, 128, 128]
        e3 = self.enc3(self.pool(e2))        # [B, 64,  64,  64]
        bottleneck = self.pool(e3)           # [B, 64,  32,  32]

        d3 = self.dec3_up(bottleneck)                        # [B, 32, 64, 64]
        d3 = self.dec3_conv(torch.cat([d3, e3], dim=1))     # [B, 32, 64, 64]

        d2 = self.dec2_up(d3)                                # [B, 16, 128, 128]
        d2 = self.dec2_conv(torch.cat([d2, e2], dim=1))     # [B, 16, 128, 128]

        d1 = self.dec1_up(d2)                                # [B,  8, 256, 256]
        d1 = self.dec1_conv(torch.cat([d1, e1], dim=1))     # [B,  8, 256, 256]

        return self.final(d1)                                # [B,  1, 256, 256]
