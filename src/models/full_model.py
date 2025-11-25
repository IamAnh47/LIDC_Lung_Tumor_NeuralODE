import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

# Import cả 2 loại Encoder để lựa chọn
from .encoders import MedicalResNetEncoder, NNUnetEncoder
from .ode_func import ODEFunc
from .decoders import ImplicitDecoder


class NeuralODE3DReconstruction(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Lấy tham số từ config
        self.latent_dim = cfg['model']['latent_dim']
        hidden_dim = cfg['model']['hidden_dim']
        roi_depth = cfg['data']['roi_size'][0]  # Số lát cắt theo trục Z (thời gian)

        # --- 1. CHỌN ENCODER ---
        enc_name = cfg['model']['encoder_name'].lower()
        print(f"🧠 Đang khởi tạo Encoder: {enc_name.upper()}")

        if enc_name == "resnet":
            # Dùng ResNet-3D (MedicalNet Pre-trained) -> Nhanh, ổn định
            self.encoder = MedicalResNetEncoder(latent_dim=self.latent_dim, pretrained=True)
        elif "unet" in enc_name:
            # Dùng U-Net (Custom) -> Chi tiết cao, cần train lâu
            self.encoder = NNUnetEncoder(latent_dim=self.latent_dim)
        else:
            raise ValueError(f"❌ Encoder '{enc_name}' không hợp lệ. Chọn 'resnet' hoặc 'unet'.")

        # --- 2. KHỞI TẠO CÁC KHỐI KHÁC ---
        self.ode_func = ODEFunc(latent_dim=self.latent_dim, hidden_dim=hidden_dim)
        self.decoder = ImplicitDecoder(latent_dim=self.latent_dim, hidden_dim=hidden_dim)

        # --- 3. CẤU HÌNH ODE SOLVER ---
        self.ode_method = cfg['model'].get('ode', {}).get('method', 'dopri5')
        self.rtol = cfg['model'].get('ode', {}).get('rtol', 1e-3)
        self.atol = cfg['model'].get('ode', {}).get('atol', 1e-3)

        # Tạo lưới thời gian cố định (Fixed Time Grid) để giải ODE 1 lần dùng cho cả batch
        # Từ t=0 đến t=1, số bước chia = độ sâu của ảnh ROI
        self.n_time_steps = roi_depth
        self.register_buffer('fixed_time_grid', torch.linspace(0, 1, self.n_time_steps))

    def manual_time_interpolation(self, z_grid, query_t):
        """
        Hàm nội suy tuyến tính thủ công (Manual Linear Interpolation).
        Thay thế cho F.grid_sample để tránh lỗi 'derivative not implemented' trên GPU đời mới.

        Args:
            z_grid: (Batch, Latent, T_steps) - Kết quả giải ODE
            query_t: (Batch, N_points) - Thời gian t (trục Z) của các điểm cần query [0, 1]
        Returns:
            z_interp: (Batch, N_points, Latent)
        """
        batch_size, latent_dim, t_steps = z_grid.shape
        _, num_points = query_t.shape

        # 1. Quy đổi thời gian thực [0, 1] sang chỉ số mảng [0, T-1]
        grid_idx = query_t * (t_steps - 1)
        # Kẹp giá trị để không bị index out of bounds (tránh lỗi CUDA)
        grid_idx = torch.clamp(grid_idx, 0, t_steps - 1 - 1e-5)

        # 2. Tìm chỉ số Sàn (Floor) và Trần (Ceil)
        idx_floor = torch.floor(grid_idx).long()
        idx_ceil = idx_floor + 1

        # 3. Tính trọng số nội suy (Khoảng cách từ sàn đến điểm thực)
        # w = 0 -> Lấy giá trị tại Floor; w = 1 -> Lấy giá trị tại Ceil
        weight = grid_idx - idx_floor.float()  # (B, N)
        weight = weight.unsqueeze(1)  # (B, 1, N) để broadcast

        # 4. Lấy giá trị Latent tại Floor và Ceil
        # Mở rộng index để khớp với chiều Latent: (B, N) -> (B, Latent, N)
        idx_floor_expanded = idx_floor.unsqueeze(1).expand(-1, latent_dim, -1)
        idx_ceil_expanded = idx_ceil.unsqueeze(1).expand(-1, latent_dim, -1)

        # Gather: Lấy vector z tại các chỉ số thời gian tương ứng
        z_floor = torch.gather(z_grid, 2, idx_floor_expanded)
        z_ceil = torch.gather(z_grid, 2, idx_ceil_expanded)

        # 5. Công thức nội suy: (1-w)*a + w*b
        z_interp = (1 - weight) * z_floor + weight * z_ceil

        # Đảo chiều về (Batch, N, Latent) để đưa vào Decoder
        return z_interp.permute(0, 2, 1)

    def forward(self, roi_image, query_coords):
        """
        Luồng xử lý chính (Forward Pass).
        Input: Ảnh CT + Tọa độ điểm (x,y,z)
        Output: Giá trị SDF dự đoán
        """
        # --- BƯỚC 1: ENCODE ---
        # Trích xuất đặc trưng hình dạng cơ bản từ ảnh 3D
        z0 = self.encoder(roi_image)  # (Batch, Latent)

        # --- BƯỚC 2: GIẢI ODE (DYNAMICS) ---
        # Tính toán sự biến đổi hình dạng dọc theo trục thời gian (Z)
        z_grid = odeint(
            self.ode_func,
            z0,
            self.fixed_time_grid,
            method=self.ode_method,
            rtol=self.rtol,
            atol=self.atol
        )
        # Output gốc của odeint là (T, B, L), đảo lại thành (B, L, T) cho dễ xử lý
        z_grid = z_grid.permute(1, 2, 0)

        # --- BƯỚC 3: NỘI SUY ĐẶC TRƯNG (INTERPOLATION) ---
        # Lấy tọa độ Z (thời gian) của các điểm cần dự đoán
        query_z = query_coords[..., 0]  # (Batch, N)

        # Nội suy để lấy vector đặc trưng chính xác tại độ sâu Z đó
        z_query = self.manual_time_interpolation(z_grid, query_z)  # (Batch, N, Latent)

        # --- BƯỚC 4: SKIP CONNECTION (RESIDUAL LEARNING) ---
        # Cộng vector gốc z0 vào vector biến đổi z_query.
        # Giúp model không bị quên thông tin gốc và hội tụ nhanh hơn.

        # Mở rộng z0: (B, Latent) -> (B, N, Latent)
        z0_expanded = z0.unsqueeze(1).expand(-1, z_query.shape[1], -1)

        # Phép cộng thần thánh
        z_combined = z_query + z0_expanded

        # --- BƯỚC 5: DECODE (IMPLICIT FUNCTION) ---
        # Lấy tọa độ không gian 2D (Y, X)
        query_xy = query_coords[..., 1:]  # (Batch, N, 2)

        # Đưa vào Decoder để ra giá trị SDF
        pred_sdf = self.decoder(z_combined, query_xy)

        return pred_sdf