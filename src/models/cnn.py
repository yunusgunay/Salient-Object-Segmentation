# src/models/cnn.py
import torch
import torch.nn as nn

class CNNAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder: Compresses the 3-channel RBG image into deep feature maps
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Resolution / 2 (128)

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Resolution / 4 (64)

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Resolution / 8 (32)
        )

        # Decoder: Expands the feature maps back to a 1-channel binary mask
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            # Final output layer (1 channel for binary mask)
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        mask_logits = self.decoder(features)
        return mask_logits
