import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class CBAM(nn.Module):
    """
    非常常見的 attention 模組：
      - Channel Attention
      - Spatial Attention
    在實驗 V (final) 會打開。
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()
        # Channel attention
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx = F.adaptive_max_pool2d(x, 1).view(b, c)
        ca = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        x = x * ca

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.cat([avg_out, max_out], dim=1)
        sa = torch.sigmoid(self.spatial(sa))
        x = x * sa
        return x


class SimpleMambaBlock(nn.Module):
    """
    這裡用「Mamba 風格」block（不依賴 mamba-ssm），
    主要目的：在 bottleneck 加入 sequence-like 全域建模。
    """
    def __init__(self, channels):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pwconv = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.pwconv(x)
        x = self.norm(x)
        x = self.act(x)
        return x + residual


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # padding 對齊
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UMambaUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, use_attention=False):
        super().__init__()
        self.use_attention = use_attention

        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = Down(64, 128)
        self.enc3 = Down(128, 256)
        self.enc4 = Down(256, 512)

        self.mamba = SimpleMambaBlock(512)

        if use_attention:
            self.cbam4 = CBAM(512)
            self.cbam3 = CBAM(256)
            self.cbam2 = CBAM(128)
            self.cbam1 = CBAM(64)
        else:
            self.cbam4 = self.cbam3 = self.cbam2 = self.cbam1 = nn.Identity()

        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)

        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # encoder
        x1 = self.enc1(x)      # 64
        x2 = self.enc2(x1)     # 128
        x3 = self.enc3(x2)     # 256
        x4 = self.enc4(x3)     # 512

        # bottleneck (Mamba-like)
        x4 = self.mamba(x4)
        x4 = self.cbam4(x4)

        # decoder
        x = self.up1(x4, self.cbam3(x3))
        x = self.up2(x,  self.cbam2(x2))
        x = self.up3(x,  self.cbam1(x1))

        logits = self.out_conv(x)
        return logits
