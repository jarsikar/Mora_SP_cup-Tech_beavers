"""
model.py - Neural Denoiser Architectures for Mora SP Cup 2026
--------------------------------------------------------------
Provides two state-of-the-art lightweight architectures:
1. DnCNN: Classic deep residual learning network (Zhang et al.)
2. LightweightUNet: Multi-scale residual encoder-decoder with skip connections

Both architectures learn the residual noise mapping:
    Predicted Clean = Input Noisy - Residual Noise
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DnCNN(nn.Module):
    """
    Lightweight Deep Convolutional Neural Network for Image Denoising (DnCNN).
    Predicts the residual noise map to be subtracted from the noisy input.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 3, num_features: int = 48, num_layers: int = 12):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=True),
            nn.PReLU(num_features),
        ]
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(num_features),
                nn.PReLU(num_features),
            ])
        layers.append(
            nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1, bias=True)
        )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual learning: clean = noisy - predicted_noise
        residual = self.network(x)
        return x - residual


class ResidualBlock(nn.Module):
    """Simple Residual Block with PReLU activation."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.PReLU(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.act(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))
        return self.act(x + res)


class LightweightUNet(nn.Module):
    """
    Compact Multi-Scale Residual UNet (~350k parameters).
    Combines multi-scale receptive field with residual skip connections.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 32):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        # Encoder
        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.PReLU(c1),
            ResidualBlock(c1),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.PReLU(c2),
            ResidualBlock(c2),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.PReLU(c3),
            ResidualBlock(c3),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(c3),
            ResidualBlock(c3),
        )

        # Decoder
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2),
            nn.PReLU(c2),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(c2 + c2, c2, kernel_size=3, padding=1),
            nn.PReLU(c2),
            ResidualBlock(c2),
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2),
            nn.PReLU(c1),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(c1 + c1, c1, kernel_size=3, padding=1),
            nn.PReLU(c1),
            ResidualBlock(c1),
        )

        self.outc = nn.Conv2d(c1, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-scale feature extraction
        e1 = self.inc(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)

        b = self.bottleneck(e3)

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        residual = self.outc(d1)
        # Residual learning
        return x - residual


def build_model(model_name: str = "unet", **kwargs) -> nn.Module:
    if model_name.lower() == "dncnn":
        return DnCNN(**kwargs)
    elif model_name.lower() in ("unet", "lightweight_unet"):
        return LightweightUNet(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose 'dncnn' or 'unet'.")


if __name__ == "__main__":
    # Quick parameter check
    m1 = DnCNN()
    m2 = LightweightUNet()
    p1 = sum(p.numel() for p in m1.parameters() if p.requires_grad)
    p2 = sum(p.numel() for p in m2.parameters() if p.requires_grad)
    print(f"DnCNN trainable parameters: {p1:,}")
    print(f"LightweightUNet trainable parameters: {p2:,}")

    dummy = torch.randn(2, 3, 128, 128)
    out1 = m1(dummy)
    out2 = m2(dummy)
    assert out1.shape == dummy.shape
    assert out2.shape == dummy.shape
    print("Shape test passed successfully!")
