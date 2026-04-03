# src/models/dino.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class DINOSegmenter(nn.Module):
    def __init__(
        self,
        model_name: str = "vit_small_patch16_224.dino",
        out_channels: int = 1,
        freeze_encoder: bool = True,
    ):
        super().__init__()

        self.encoder = timm.create_model(model_name, pretrained=True, num_classes=0)

        if not hasattr(self.encoder, "patch_embed"):
            raise ValueError(f"Model {model_name} does not expose patch embedding as expected.")
        
        self.patch_size = self.encoder.patch_embed.patch_size[0]
        self.embed_dim = self.encoder.num_features

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Decoder: 14x14 -> 224x224
        self.decoder = nn.Sequential(
            nn.Conv2d(self.embed_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, out_channels, kernel_size=1)
        )
    
    def forward(self, x):
        tokens = self.encoder.forward_features(x)

        if isinstance(tokens, (list, tuple)):
            tokens = tokens[-1]
        
        if tokens.dim() != 3:
            raise ValueError(f"Expected [B, N, C] or [B, N+1, C], got {tokens.shape}")

        b, n, c = tokens.shape
    
        # Remove [CLS] token if present
        if n == 197:
            tokens = tokens[:, 1:, :]
            n -= 1
        elif n != 196:
            raise ValueError(f"Expected [B, N, C] or [B, N+1, C], got {tokens.shape}")
        
        h = w = int(n**0.5)
        x = tokens.transpose(1, 2).reshape(b, c, h, w)

        x = self.decoder(x)

        # Safety check for output size
        if x.shape[2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    
        return x
