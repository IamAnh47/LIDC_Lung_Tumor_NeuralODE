import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint

# --- ĐỔI IMPORT TẠI ĐÂY ---
# from .encoders import NNUnetEncoder
from .encoders import MedicalResNetEncoder # <-- Dùng cái này
# --------------------------

from .ode_func import ODEFunc
from .decoders import ImplicitDecoder


class NeuralODE3DReconstruction(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.latent_dim = cfg['model']['latent_dim']
        hidden_dim = cfg['model']['hidden_dim']
        roi_depth = cfg['data']['roi_size'][0]

        # --- ĐỔI MODEL TẠI ĐÂY ---
        # self.encoder = NNUnetEncoder(latent_dim=self.latent_dim)
        self.encoder = MedicalResNetEncoder(latent_dim=self.latent_dim)
        # -------------------------

        self.ode_func = ODEFunc(latent_dim=self.latent_dim, hidden_dim=hidden_dim)
        self.decoder = ImplicitDecoder(latent_dim=self.latent_dim, hidden_dim=hidden_dim)

        self.ode_method = cfg['model'].get('ode_method', 'dopri5')
        self.rtol = cfg['model'].get('ode', {}).get('rtol', 1e-3)
        self.atol = cfg['model'].get('ode', {}).get('atol', 1e-3)

        self.n_time_steps = roi_depth
        self.register_buffer('fixed_time_grid', torch.linspace(0, 1, self.n_time_steps))

    def manual_time_interpolation(self, z_grid, query_t):
        """
        Thay thế F.grid_sample bằng nội suy thủ công để tránh lỗi CUDA.
        """
        batch_size, latent_dim, t_steps = z_grid.shape
        _, num_points = query_t.shape

        # 1. Quy đổi thời gian [0, 1] sang chỉ số [0, T-1]
        grid_idx = query_t * (t_steps - 1)
        grid_idx = torch.clamp(grid_idx, 0, t_steps - 1 - 1e-5)

        # 2. Tìm cận dưới (floor) và cận trên (ceil)
        idx_floor = torch.floor(grid_idx).long()
        idx_ceil = idx_floor + 1

        # 3. Tính trọng số nội suy
        weight = grid_idx - idx_floor.float()  # (B, N)
        weight = weight.unsqueeze(1)  # (B, 1, N)

        # 4. Gather values
        idx_floor_expanded = idx_floor.unsqueeze(1).expand(-1, latent_dim, -1)
        idx_ceil_expanded = idx_ceil.unsqueeze(1).expand(-1, latent_dim, -1)

        z_floor = torch.gather(z_grid, 2, idx_floor_expanded)
        z_ceil = torch.gather(z_grid, 2, idx_ceil_expanded)

        # 5. Nội suy tuyến tính
        z_interp = (1 - weight) * z_floor + weight * z_ceil

        # Đảo chiều về (B, N, Latent)
        return z_interp.permute(0, 2, 1)

    def forward(self, roi_image, query_coords):
        # --- BƯỚC 1: ENCODE ---
        z0 = self.encoder(roi_image)  # (Batch, Latent)

        # --- BƯỚC 2: GIẢI ODE TRÊN LƯỚI CỐ ĐỊNH ---
        z_grid = odeint(
            self.ode_func,
            z0,
            self.fixed_time_grid,
            method=self.ode_method,
            rtol=self.rtol,
            atol=self.atol
        )
        # (T, B, L) -> (B, L, T)
        z_grid = z_grid.permute(1, 2, 0)

        # --- BƯỚC 3: NỘI SUY ---
        # Lấy tọa độ Z (thời gian)
        query_z = query_coords[..., 0]  # (B, N)

        # Dùng hàm thủ công
        z_query = self.manual_time_interpolation(z_grid, query_z)  # (B, N, Latent)

        # --- BƯỚC 4: SKIP CONNECTION (QUAN TRỌNG) ---
        # Cộng vector gốc z0 vào vector biến đổi z_query
        # Giúp model học dễ hơn (Residual Learning)

        # Expand z0 từ (B, Latent) -> (B, N, Latent) để cộng được
        z0_expanded = z0.unsqueeze(1).expand(-1, z_query.shape[1], -1)

        # Kết hợp
        z_combined = z_query + z0_expanded

        # --- BƯỚC 5: DECODE ---
        query_xy = query_coords[..., 1:]
        pred_sdf = self.decoder(z_combined, query_xy)

        return pred_sdf