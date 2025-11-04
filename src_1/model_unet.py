# model_unet.py
import torch
import torch.nn as nn
from torchvision import models

class UNetBinary(nn.Module):
    """
    Binary segmentation UNet-like using ResNet34 encoder.
    Output: single channel logits (B,1,H,W). Use BCEWithLogitsLoss.
    """
    def __init__(self, dropout_p=0.2, pretrained=True):
        super().__init__()
        backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1) if pretrained else models.resnet34(weights=None)

        # Encoder parts
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # /2
        self.pool = backbone.maxpool                                          # /4
        self.enc1 = backbone.layer1  # /4
        self.enc2 = backbone.layer2  # /8
        self.enc3 = backbone.layer3  # /16
        self.enc4 = backbone.layer4  # /32

        # Decoder blocks
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # /16
        self.dec4 = self.conv_block(512, 256, dropout_p)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # /8
        self.dec3 = self.conv_block(256, 128, dropout_p)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)   # /4
        self.dec2 = self.conv_block(128, 64, dropout_p)

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)    # /2
        self.dec1 = self.conv_block(128, 64, dropout_p)

        # extra up to full res if needed
        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)    # /1
        self.dec0 = self.conv_block(32 + 3, 32, dropout_p)  # cat with stem features if resized

        self.final = nn.Conv2d(32, 1, kernel_size=1)  # logits

    def conv_block(self, in_ch, out_ch, dropout_p):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p)
        )

    def _pad_to(self, src, tgt):
        # pad src to match tgt spatial dims (centered)
        sh, sw = src.shape[2], src.shape[3]
        th, tw = tgt.shape[2], tgt.shape[3]
        pad_h = max(0, th - sh)
        pad_w = max(0, tw - sw)
        if pad_h == 0 and pad_w == 0:
            return src
        pad = (pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2)
        return nn.functional.pad(src, pad)

    def forward(self, x):
        # Encoder
        s = self.stem(x)            # [B,64,H/2,W/2]
        p = self.pool(s)           # [B,64,H/4,W/4]
        e1 = self.enc1(p)          # [B,64,H/4,W/4]
        e2 = self.enc2(e1)         # [B,128,H/8,W/8]
        e3 = self.enc3(e2)         # [B,256,H/16,W/16]
        e4 = self.enc4(e3)         # [B,512,H/32,W/32]

        # Decoder
        d4 = self.up4(e4)
        if d4.shape[2:] != e3.shape[2:]:
            d4 = self._pad_to(d4, e3)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        if d3.shape[2:] != e2.shape[2:]:
            d3 = self._pad_to(d3, e2)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        if d2.shape[2:] != e1.shape[2:]:
            d2 = self._pad_to(d2, e1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        if d1.shape[2:] != s.shape[2:]:
            # pad or crop s to match
            s_for_cat = s
            if s_for_cat.shape[2:] != d1.shape[2:]:
                s_for_cat = self._pad_to(s_for_cat, d1)
        else:
            s_for_cat = s
        d1 = torch.cat([d1, s_for_cat], dim=1)
        d1 = self.dec1(d1)

        d0 = self.up0(d1)
        # if input size not exactly handled, pad to x
        if d0.shape[2:] != x.shape[2:]:
            d0 = self._pad_to(d0, x)
        # combine shallow features (upsampled) with original image downsampled if needed
        # to keep channels aligned, we'll just pass d0 forward
        d0 = self.dec0(torch.cat([d0, nn.functional.interpolate(x, size=d0.shape[2:], mode='bilinear', align_corners=False)], dim=1))

        out = self.final(d0)  # logits, shape [B,1,H,W]
        return out
