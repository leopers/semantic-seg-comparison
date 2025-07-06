import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Applies two consecutive 3x3 convolutions + ReLU.
    Preserves spatial size (padding=1).
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownSample(nn.Module):
    """
    Downsampling block with max pooling + DoubleConv.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.MaxPool2d(kernel_size=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.down(x)
        return self.conv(x)


class UpSample(nn.Module):
    """
    Upsampling block with ConvTranspose2d + DoubleConv.
    """

    def __init__(self, up_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(up_channels, up_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(up_channels + skip_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        if x1.shape[-2:] != x2.shape[-2:]:
            x1 = F.interpolate(
                x1, size=x2.shape[-2:], mode="bilinear", align_corners=False
            )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Lightweight U-Net architecture for binary segmentation on 128x128 images.
    Encoder depth: 3 levels. Channels: 16 → 32 → 64.
    Decoder depth: 3 levels. Channels: 64 → 32 → 16.
    Final output: 1 channel (binary mask).
    """

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_channels, 16)
        self.enc2 = DownSample(16, 32)
        self.enc3 = DownSample(32, 64)

        # Decoder
        self.up3 = UpSample(64, 32, 32)
        self.up2 = UpSample(32, 16, 16)

        # Final output
        self.outc = nn.Conv2d(16, out_channels, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)

        # Decoder
        dec3 = self.up3(enc3, enc2)
        dec2 = self.up2(dec3, enc1)

        # Final output
        out = self.outc(dec2)
        return self.activation(out)
