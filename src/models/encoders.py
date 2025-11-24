import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


# ==============================================================================
# 1. OPTION A: RESNET-3D (RECOMMENDED FOR CONVERGENCE)
# Sử dụng Pre-trained Weights từ Kinetics-400 giúp model hội tụ cực nhanh
# ==============================================================================
class MedicalResNetEncoder(nn.Module):
    def __init__(self, input_dim=1, latent_dim=256):
        super().__init__()

        # Load ResNet-18 3D với trọng số đã train sẵn (Kinetics-400)
        # weights='DEFAULT' tương đương với trọng số tốt nhất hiện có
        self.backbone = r3d_18(weights='DEFAULT')

        # --- SỬA LỚP ĐẦU VÀO (Input Layer) ---
        # ResNet gốc nhận video RGB (3 kênh). Ảnh CT chỉ có 1 kênh (Grayscale).
        # Ta thay Conv3d(3, ...) bằng Conv3d(1, ...)
        old_conv = self.backbone.stem[0]

        new_conv = nn.Conv3d(
            in_channels=input_dim,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        # Mẹo Transfer Learning: Lấy trung bình cộng trọng số của 3 kênh RGB
        # gán vào kênh duy nhất của Conv mới để giữ lại kiến thức đã học.
        with torch.no_grad():
            new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)

        self.backbone.stem[0] = new_conv

        # --- SỬA LỚP ĐẦU RA (Output Layer) ---
        # Thay lớp FC cuối cùng (400 class video) bằng lớp FC ra latent_dim
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, latent_dim)

        # Activation cuối cùng
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        # x: (Batch, 1, D, H, W)
        # ResNet3D tự động xử lý feature extraction và flatten
        z0 = self.backbone(x)  # -> (Batch, latent_dim)
        z0 = self.act(z0)
        return z0

# ==============================================================================
# 2. OPTION B: NNUNET ENCODER (LƯU ĐỂ DÙNG SAU)
# Kiến trúc U-Net thuần, tốt cho chi tiết nhưng khó train từ đầu
# ==============================================================================
# from monai.networks.blocks import Convolution
#
# class NNUnetEncoder(nn.Module):
#     def __init__(self, input_dim=1, latent_dim=256):
#         super().__init__()
#         self.blocks = nn.ModuleList([
#             Convolution(spatial_dims=3, in_channels=input_dim, out_channels=32, strides=2, kernel_size=3, padding=1, norm="INSTANCE", act=("LEAKYRELU", {"negative_slope": 0.01})),
#             Convolution(spatial_dims=3, in_channels=32, out_channels=64, strides=2, kernel_size=3, padding=1, norm="INSTANCE", act=("LEAKYRELU", {"negative_slope": 0.01})),
#             Convolution(spatial_dims=3, in_channels=64, out_channels=128, strides=2, kernel_size=3, padding=1, norm="INSTANCE", act=("LEAKYRELU", {"negative_slope": 0.01})),
#             Convolution(spatial_dims=3, in_channels=128, out_channels=256, strides=2, kernel_size=3, padding=1, norm="INSTANCE", act=("LEAKYRELU", {"negative_slope": 0.01})),
#             Convolution(spatial_dims=3, in_channels=256, out_channels=512, strides=1, kernel_size=3, padding=1, norm="INSTANCE", act=("LEAKYRELU", {"negative_slope": 0.01}))
#         ])
#         self.flatten_dim = 512 * 4 * 4 * 2
#         self.fc_z0 = nn.Linear(self.flatten_dim, latent_dim)
#
#     def forward(self, x):
#         h = x
#         for block in self.blocks:
#             h = block(h)
#         h_flat = h.view(h.size(0), -1)
#         z0 = self.fc_z0(h_flat)
#         return z0