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

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔌 Device: {device}")

    # 2. Load Model & Checkpoint
    model = NeuralODE3DReconstruction(cfg).to(device)

    # checkpoint_path = "experiments/exp_01_nnunet/checkpoints/best_model.pth"
    # if not os.path.exists(checkpoint_path):
    #     print("⚠️ Chưa có best_model.pth, thử load last.pth...")
    #     checkpoint_path = "experiments/exp_01_nnunet/checkpoints/last.pth"

    checkpoint_path = "experiments/exp_02_resnet/checkpoints/best_model.pth"
    if not os.path.exists(checkpoint_path):
        print("⚠️ Chưa có best_model.pth, thử load last.pth...")
        checkpoint_path = "experiments/exp_02_resnet/checkpoints/last.pth"

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        print(f"✅ Đã load model từ: {checkpoint_path}")
    else:
        print("❌ Không tìm thấy checkpoint nào để test!")
        return

    model.eval()

    # 3. Lấy 1 mẫu từ tập Train (hoặc Val) để xem nó học được chưa
    processed_dir = cfg['paths']['processed_data']
    dataset = LIDCDataset(processed_dir, split='test')  # Test trên train cho dễ

    if len(dataset) == 0:
        print("❌ Dataset rỗng!")
        return

    # Lấy mẫu đầu tiên
    roi, points, gt_sdf = dataset[0]
    roi = roi.unsqueeze(0).to(device)  # (1, 1, D, H, W)

    print(f"🔍 Đang kiểm tra file ID: {dataset.file_ids[0]}")

    # 4. Tạo lưới điểm dày đặc để dự đoán (giống lúc inference)
    roi_size = cfg['data']['roi_size']
    z = torch.linspace(0, 1, 32)  # Giảm độ phân giải chút để chạy nhanh
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

    # Vẽ biểu đồ và lưu vào thư mục debugs
    try:
        # --- TẠO THƯ MỤC DEBUGS ---
        debug_dir = "debugs"
        os.makedirs(debug_dir, exist_ok=True)
        save_path = os.path.join(debug_dir, "debug_sdf_dist.png")
        # --------------------------

        plt.figure(figsize=(8, 5))  # Tạo figure mới để tránh vẽ chồng
        plt.hist(pred_sdf.cpu().numpy().flatten(), bins=50, color='blue', alpha=0.7)
        plt.axvline(x=0, color='red', linestyle='--', label="Bề mặt (0.0)")
        plt.title(f"Phân bố SDF - {dataset.file_ids[0]}")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig(save_path)  # Lưu vào đường dẫn mới
        print(f"🖼️ Đã lưu biểu đồ tại: {save_path}")
        plt.close()  # Đóng figure để giải phóng bộ nhớ
    except Exception as e:
        print(f"⚠️ Không thể vẽ biểu đồ: {e}")


if __name__ == "__main__":
    check_model_prediction()