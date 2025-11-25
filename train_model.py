import os
import yaml
import argparse
import torch
from torch.utils.data import DataLoader

# Import các module từ src
from src.models.full_model import NeuralODE3DReconstruction
from src.data.dataset_loader import LIDCDataset
from src.training.trainer import Trainer


def main():
    # --- 1. Cấu hình tham số (Arguments) ---
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Neural ODE cho LIDC-IDRI")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Đường dẫn file cấu hình")

    # 👇 THÊM THAM SỐ NÀY ĐỂ RESUME 👇
    parser.add_argument("--resume", type=str, default=None,
                        help="Đường dẫn file .pth để train tiếp (VD: experiments/.../last.pth)")

    args = parser.parse_args()

    # Load Config từ file YAML
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"❌ Không tìm thấy file config tại: {args.config}")

    # --- FIX LỖI UNICODE TRÊN WINDOWS ---
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print(f"🚀 BẮT ĐẦU TRAINING: {cfg['project']['name']}")
    print("-" * 50)

    # --- 2. Thiết lập thiết bị (Device) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔌 Thiết bị sử dụng: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # --- 3. Chuẩn bị Dữ liệu (Data Loaders) ---
    processed_dir = os.path.abspath(cfg['paths']['processed_data'])

    train_dataset = LIDCDataset(processed_dir, split='train')

    # --- DEBUG MODE (Đã comment lại) ---
    # from torch.utils.data import Subset
    # train_dataset = Subset(train_dataset, [10])
    # print("⚠️ ĐANG CHẠY CHẾ ĐỘ DEBUG: CHỈ TRAIN 1 MẪU!")
    # ------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_dataset = LIDCDataset(processed_dir, split='val')
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['train']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"📦 Dữ liệu Train: {len(train_dataset)} mẫu")
    print(f"📦 Dữ liệu Val:   {len(val_dataset)} mẫu")

    # --- 4. Khởi tạo Mô hình (Model) ---
    print("🧠 Đang khởi tạo mô hình Neural ODE + ResNet-3D-MedicalNet Encoder...")
    model = NeuralODE3DReconstruction(cfg)

    # --- 5. Khởi tạo Trainer ---
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        device=device
    )

    # --- 👇 LOGIC LOAD CHECKPOINT ĐỂ CHẠY TIẾP 👇 ---
    start_epoch = 1
    best_val_loss = float('inf')

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"🔄 Đang khôi phục training từ: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)

            # 1. Load trọng số Model
            model.load_state_dict(checkpoint['state_dict'])

            # 2. Load trạng thái Optimizer (Để giữ Learning Rate đang chạy dở)
            if 'optimizer' in checkpoint:
                trainer.optimizer.load_state_dict(checkpoint['optimizer'])

            # 3. Cập nhật Epoch bắt đầu
            start_epoch = checkpoint['epoch'] + 1
            print(f"✅ Khôi phục thành công! Sẽ bắt đầu từ Epoch {start_epoch}")
        else:
            print(f"⚠️ Không tìm thấy file checkpoint: {args.resume}. Sẽ train từ đầu.")
    # -------------------------------------------------

    # --- 6. Vòng lặp Huấn luyện (Training Loop) ---
    print("🔥 Bắt đầu vòng lặp huấn luyện...")

    # Sửa range để chạy từ start_epoch
    for epoch in range(start_epoch, cfg['train']['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{cfg['train']['num_epochs']}")

        # Train 1 epoch
        train_loss = trainer.train_epoch(epoch)
        print(f"   📉 Train Loss: {train_loss:.6f}")

        # Validate định kỳ
        if epoch % cfg['train'].get('val_every', 1) == 0:
            val_loss = trainer.validate(epoch)
            print(f"   🔍 Val Loss:   {val_loss:.6f}")

            # Lưu model tốt nhất (Checkpointing)
            if val_loss < best_val_loss:
                print(f"   ⭐ Loss giảm ({best_val_loss:.6f} -> {val_loss:.6f}). Đang lưu Best Model...")
                best_val_loss = val_loss
                trainer.save_checkpoint(epoch, is_best=True)

        # Lưu model định kỳ (để resume nếu sập nguồn)
        if epoch % cfg['train']['save_every'] == 0:
            trainer.save_checkpoint(epoch, is_best=False)
            print(f"   💾 Đã lưu checkpoint định kỳ tại Epoch {epoch}")

    print("\n✅ HUẤN LUYỆN HOÀN TẤT!")

    # Tự động xác định đường dẫn lưu dựa trên config (Hỗ trợ cả nnunet và resnet)
    current_exp = os.path.basename(cfg['paths']['experiment_dir'])
    # Mặc định trả về đường dẫn giả định nếu biến experiment_dir chỉ là thư mục cha
    # Thực tế trong trainer.py nó lưu vào ckpt_dir cụ thể
    ckpt_path = os.path.join(trainer.ckpt_dir, 'best_model.pth')
    print(f"👉 Model tốt nhất: {ckpt_path}")


if __name__ == "__main__":
    main()