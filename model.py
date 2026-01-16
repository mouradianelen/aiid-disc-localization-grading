import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h + self.skip(x))


class AttentionGate(nn.Module):
    def __init__(self, g_ch, x_ch, inter_ch):
        super().__init__()
        self.Wg = nn.Conv2d(g_ch, inter_ch, 1)
        self.Wx = nn.Conv2d(x_ch, inter_ch, 1)
        self.psi = nn.Conv2d(inter_ch, 1, 1)

    def forward(self, g, x):
        psi = torch.sigmoid(self.psi(F.relu(self.Wg(g) + self.Wx(x))))
        return x * psi


class DiscUNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.e1 = ResBlock(1, 32)
        self.e2 = ResBlock(32, 64)
        self.e3 = ResBlock(64, 128)
        self.e4 = ResBlock(128, 256)

        self.bottleneck = ResBlock(256, 256)

        self.up4 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.att4 = AttentionGate(128, 128, 64)
        self.d4 = ResBlock(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.att3 = AttentionGate(64, 64, 32)
        self.d3 = ResBlock(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.att2 = AttentionGate(32, 32, 16)
        self.d2 = ResBlock(64, 32)

        self.out = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.avg_pool2d(e1, 2))
        e3 = self.e3(F.avg_pool2d(e2, 2))
        e4 = self.e4(F.avg_pool2d(e3, 2))

        b = self.bottleneck(e4)

        d4 = self.up4(b)
        e3 = self.att4(d4, e3)
        d4 = self.d4(torch.cat([d4, e3], 1))

        d3 = self.up3(d4)
        e2 = self.att3(d3, e2)
        d3 = self.d3(torch.cat([d3, e2], 1))

        d2 = self.up2(d3)
        e1 = self.att2(d2, e1)
        d2 = self.d2(torch.cat([d2, e1], 1))

        return self.out(d2)

class DiscGraderResNet(nn.Module):
    def __init__(self, n_classes=5, pretrained=True):
        super().__init__()

        self.backbone = models.resnet18(pretrained=pretrained)

        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        self.backbone.fc = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.backbone(x)
