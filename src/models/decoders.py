import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):  # <--- PHẢI CÓ (nn.Module)
    def __init__(self, num_freqs=6, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.freq_bands = 2.0 ** torch.linspace(0.0, num_freqs - 1, num_freqs)

    def forward(self, x):
        embed = [x] if self.include_input else []
        for freq in self.freq_bands.to(x.device):
            embed.append(torch.sin(x * freq * np.pi))
            embed.append(torch.cos(x * freq * np.pi))
        return torch.cat(embed, dim=-1)


class ImplicitDecoder(nn.Module):  # <--- PHẢI CÓ (nn.Module)
    def __init__(self, latent_dim=256, hidden_dim=512):
        super().__init__()  # <--- PHẢI CÓ

        # Cấu hình: Freq=8 cho chi tiết cao
        FREQ_NUM = 8
        self.pos_enc = PositionalEncoding(num_freqs=FREQ_NUM)

        coord_dim = 2 + 2 * 2 * FREQ_NUM
        input_dim = latent_dim + coord_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Softplus(beta=100),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(beta=100),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(beta=100),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z_t, coords_2d):
        coords_encoded = self.pos_enc(coords_2d)
        inp = torch.cat([z_t, coords_encoded], dim=-1)
        sdf = self.net(inp)
        return sdf