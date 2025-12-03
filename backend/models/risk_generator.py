import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv(x)


class PestRiskGenerator(nn.Module):
    """
    UNet-like network to produce a pest/disease risk heatmap from LAI/NDVI.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.down = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        self.out = nn.Conv2d(base_channels, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        d1 = self.down(e1)
        e2 = self.enc2(d1)
        d2 = self.down(e2)
        e3 = self.enc3(d2)
        d3 = self.down(e3)

        b = self.bottleneck(d3)

        u3 = self.up3(b)
        c3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(c3)

        u2 = self.up2(d3)
        c2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(c2)

        u1 = self.up1(d2)
        c1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(c1)

        out = torch.sigmoid(self.out(d1))
        return out
