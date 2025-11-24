import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    """
    Biến đổi tọa độ (x, y) thành các đặc trưng tần số cao.
    Giúp MLP học được các chi tiết hình học sắc nét hơn.
    """

    def __init__(self, num_freqs=6, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        # Tạo các tần số: 2^0, 2^1, ..., 2^(N-1)
        self.freq_bands = 2.0 ** torch.linspace(0.0, num_freqs - 1, num_freqs)

    def forward(self, x):
        # x shape: (B, N, 2) -> tọa độ y, x
        embed = [x] if self.include_input else []

        for freq in self.freq_bands.to(x.device):
            embed.append(torch.sin(x * freq * np.pi))
            embed.append(torch.cos(x * freq * np.pi))

        # Nối lại: (B, N, 2 + 2 * 2 * num_freqs)
        return torch.cat(embed, dim=-1)


class ImplicitDecoder(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=512):
        super().__init__()

        # --- CẤU HÌNH POSITIONAL ENCODING ---
        self.pos_enc = PositionalEncoding(num_freqs=4)

        # Tính toán kích thước đầu vào mới
        # Tọa độ gốc (2 chiều: y, x) -> Qua Encoding sẽ tăng lên
        # Công thức: input_dim + input_dim * 2 * num_freqs
        # Với 2 chiều, num_freqs=6 -> 2 + 2*2*6 = 26 chiều
        coord_dim = 2 + 2 * 2 * 4

        # Input của mạng = Latent z + Encoded Coords
        input_dim = latent_dim + coord_dim

        # --- MẠNG MLP (Sử dụng Softplus thay vì ReLU để mượt hơn) ---
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Softplus(beta=100),  # Softplus tốt hơn ReLU cho SDF

            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(beta=100),

            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(beta=100),

            nn.Linear(hidden_dim, 1)  # Output SDF
        )

    def forward(self, z_t, coords_2d):
        # 1. Mã hóa tọa độ không gian
        coords_encoded = self.pos_enc(coords_2d)

        # 2. Ghép với Latent Vector
        # z_t: (B, N, Latent)
        # coords_encoded: (B, N, Encoded_Dim)
        inp = torch.cat([z_t, coords_encoded], dim=-1)

        # 3. Dự đoán
        sdf = self.net(inp)
        return sdf