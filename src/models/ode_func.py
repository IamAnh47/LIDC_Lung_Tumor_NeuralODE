import torch
import torch.nn as nn

class ODEFunc(nn.Module):
    """
    Hàm f(z, t) mô tả tốc độ biến đổi của hình dạng khối u dọc theo trục Z (thời gian t).
    """
    def __init__(self, latent_dim=256, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            # Input: vector z + giá trị thời gian t
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.Tanh(), # Tanh thường ổn định hơn ReLU cho ODE
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim) # Output: dz/dt (cùng chiều với z)
        )

    def forward(self, t, z):
        # Neural ODE solver truyền t là một scalar, ta cần broadcast nó ra
        # z shape: (Batch, Latent)
        # t shape: scalar tensor
        
        t_vec = torch.ones(z.shape[0], 1).to(z.device) * t
        
        # Ghép [z, t] để mạng học được sự phụ thuộc vào độ sâu
        z_input = torch.cat([z, t_vec], dim=1)
        
        # Tính đạo hàm
        dz_dt = self.net(z_input)
        return dz_dt