# model_unet.py
import torch
import torch.nn as nn
from torchvision import models

class UNet(nn.Module):
    def __init__(self, n_classes=13):
        super().__init__()

        # Encoder backbone (ResNet34 pretrained)
        backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])  # output: (B, 512, H/32, W/32)

        # Decoder path (mirrors the ResNet34 downsamples)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 2, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 2, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(16, n_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
