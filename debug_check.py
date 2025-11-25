import torch
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
from src.models.full_model import NeuralODE3DReconstruction
from src.data.dataset_loader import LIDCDataset


def check_model_prediction():
    # 1. Load Config
    config_path = "configs/config.yaml"
    if not os.path.exists(config_path):
        print("❌ Không thấy file config!")
        return

    # --- FIX LỖI UNICODE: Thêm encoding="utf-8" ---
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # ----------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔌 Device: {device}")

    # 2. Load Model & Checkpoint
    model = NeuralODE3DReconstruction(cfg).to(device)

    # Tự động chọn đường dẫn checkpoint dựa trên config (ResNet hay UNet)
    enc_name = cfg['model']['encoder_name']
    if enc_name == "resnet":
        exp_name = "exp_02_resnet"
    else:
        exp_name = "exp_01_unet"  # Hoặc tên khác nếu đổi config

    base_ckpt = os.path.join("experiments", exp_name, "checkpoints")
    checkpoint_path = os.path.join(base_ckpt, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print(f"⚠️ Chưa có best_model.pth tại {checkpoint_path}, thử load last.pth...")
        checkpoint_path = os.path.join(base_ckpt, "last.pth")

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        print(f"✅ Đã load model từ: {checkpoint_path}")
    else:
        print(f"❌ Không tìm thấy checkpoint nào tại {base_ckpt}!")
        return

    model.eval()

    # 3. Lấy 1 mẫu từ tập Test
    processed_dir = cfg['paths']['processed_data']
    dataset = LIDCDataset(processed_dir, split='test')  # Kiểm tra trên tập Test

    if len(dataset) == 0:
        print("❌ Dataset rỗng!")
        return

    # Lấy mẫu đầu tiên
    roi, points, gt_sdf = dataset[0]
    roi = roi.unsqueeze(0).to(device)

    print(f"🔍 Đang kiểm tra file ID: {dataset.file_ids[0]}")

    # 4. Tạo lưới điểm
    roi_size = cfg['data']['roi_size']
    z = torch.linspace(0, 1, 32)
    y = torch.linspace(0, 1, 64)
    x = torch.linspace(0, 1, 64)
    grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')
    query_coords = torch.stack([grid_z, grid_y, grid_x], dim=-1).reshape(-1, 3).unsqueeze(0).to(device)

    # 5. Dự đoán
    with torch.no_grad():
        pred_sdf = model(roi, query_coords)

    # 6. Phân tích kết quả
    min_val = pred_sdf.min().item()
    max_val = pred_sdf.max().item()
    mean_val = pred_sdf.mean().item()

    print("-" * 30)
    print(f"📊 THỐNG KÊ GIÁ TRỊ SDF DỰ ĐOÁN:")
    print(f"   Min : {min_val:.4f} (Kỳ vọng < 0)")
    print(f"   Max : {max_val:.4f} (Kỳ vọng > 0)")
    print(f"   Mean: {mean_val:.4f}")
    print("-" * 30)

    if min_val > 0:
        print("❌ KẾT LUẬN: Mô hình đoán toàn bộ là 'Bên Ngoài' (Dương). Chưa tạo được Mesh.")
    else:
        print("✅ KẾT LUẬN: Mô hình đã có vùng âm! Có thể tạo được Mesh.")

    # Vẽ biểu đồ
    try:
        debug_dir = "debugs"
        os.makedirs(debug_dir, exist_ok=True)
        save_path = os.path.join(debug_dir, "debug_sdf_dist.png")

        plt.figure(figsize=(8, 5))
        plt.hist(pred_sdf.cpu().numpy().flatten(), bins=50, color='blue', alpha=0.7)
        plt.axvline(x=0, color='red', linestyle='--', label="Bề mặt (0.0)")
        plt.title(f"Phân bố SDF - {dataset.file_ids[0]}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path)
        print(f"🖼️ Đã lưu biểu đồ tại: {save_path}")
        plt.close()
    except Exception as e:
        print(f"⚠️ Không thể vẽ biểu đồ: {e}")


if __name__ == "__main__":
    check_model_prediction()