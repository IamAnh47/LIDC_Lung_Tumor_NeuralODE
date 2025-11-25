import torch
import torch.nn as nn
from torchvision.models.video import r3d_18
from monai.networks.blocks import Convolution


# ==============================================================================
# 1. RESNET-3D ENCODER
# ==============================================================================
class MedicalResNetEncoder(nn.Module):  # <--- PHẢI CÓ (nn.Module)
    def __init__(self, input_dim=1, latent_dim=256, pretrained=True):
        super().__init__()  # <--- PHẢI CÓ dòng này

        weights = 'DEFAULT' if pretrained else None
        self.backbone = r3d_18(weights=weights)

        old_conv = self.backbone.stem[0]
        new_conv = nn.Conv3d(
            in_channels=input_dim,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        if pretrained:
            with torch.no_grad():
                new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)

        self.backbone.stem[0] = new_conv
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, latent_dim)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        z0 = self.backbone(x)
        z0 = self.act(z0)
        return z0


# ==============================================================================
# 2. NN-UNET ENCODER (Optimized)
# ==============================================================================
class NNUnetEncoder(nn.Module):  # <--- PHẢI CÓ (nn.Module)
    def __init__(self, input_dim=1, latent_dim=256):
        super().__init__()  # <--- PHẢI CÓ

        self.blocks = nn.ModuleList([
            Convolution(3, input_dim, 32, strides=2, kernel_size=3, padding=1, norm="INSTANCE",
                        act=("LEAKYRELU", {"negative_slope": 0.01})),
            Convolution(3, 32, 64, strides=2, kernel_size=3, padding=1, norm="INSTANCE",
                        act=("LEAKYRELU", {"negative_slope": 0.01})),
            Convolution(3, 64, 128, strides=2, kernel_size=3, padding=1, norm="INSTANCE",
                        act=("LEAKYRELU", {"negative_slope": 0.01})),
            Convolution(3, 128, 256, strides=2, kernel_size=3, padding=1, norm="INSTANCE",
                        act=("LEAKYRELU", {"negative_slope": 0.01})),
            Convolution(3, 256, 512, strides=1, kernel_size=3, padding=1, norm="INSTANCE",
                        act=("LEAKYRELU", {"negative_slope": 0.01}))
        ])

        # Global Pooling + Dropout
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout3d(0.3)

        self.fc_z0 = nn.Linear(512, latent_dim)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = x
        for block in self.blocks:
            h = block(h)

        h = self.global_pool(h)
        h_flat = h.view(h.size(0), -1)
        h_flat = self.dropout(h_flat)

        z0 = self.fc_z0(h_flat)
        z0 = self.act(z0)
        return z0