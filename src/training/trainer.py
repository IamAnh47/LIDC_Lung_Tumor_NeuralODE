import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import numpy as np

from .losses import sdf_loss, eikonal_loss


class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config
        self.device = device

        # --- 1. CẤU HÌNH OPTIMIZER ---
        wd = float(config['train'].get('weight_decay', 0.0))
        lr = float(config['train']['learning_rate'])

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=wd
        )

        # --- 2. SCHEDULER ---
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        # --- 3. MIXED PRECISION ---
        self.scaler = torch.amp.GradScaler('cuda')

        # --- 4. LOGGING (TỰ ĐỘNG CHỌN FOLDER THEO MODEL) ---
        base_exp_dir = config['paths']['experiment_dir']  # "experiments"
        encoder_name = config['model']['encoder_name'].lower()

        # Logic đặt tên folder
        if "resnet" in encoder_name:
            exp_name = "exp_02_resnet"
        elif "unet" in encoder_name:  # Bắt cả "unet" và "nnunet"
            exp_name = "exp_01_unet"
        else:
            exp_name = f"exp_custom_{encoder_name}"

        # Đường dẫn đầy đủ: experiments/exp_02_resnet/...
        full_exp_path = os.path.join(base_exp_dir, exp_name)

        print(f"📂 Kết quả sẽ được lưu tại: {full_exp_path}")

        self.writer = SummaryWriter(log_dir=os.path.join(full_exp_path, "logs"))
        self.ckpt_dir = os.path.join(full_exp_path, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_recon = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch_idx, (roi, coords, gt_sdf) in enumerate(pbar):
            roi = roi.to(self.device)
            coords = coords.to(self.device)
            gt_sdf = gt_sdf.to(self.device)

            coords.requires_grad_(True)

            self.optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                pred_sdf = self.model(roi, coords)

                loss_recon = sdf_loss(pred_sdf, gt_sdf)
                loss_eik = eikonal_loss(pred_sdf, coords)

                # Tổng hợp Loss
                loss = loss_recon + 0.005 * loss_eik

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_val = loss.item()
            recon_val = loss_recon.item()
            total_loss += loss_val
            total_recon += recon_val

            pbar.set_postfix({'Loss': f"{loss_val:.4f}", 'Recon': f"{recon_val:.4f}"})

            global_step = (epoch - 1) * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/Loss_Total', loss_val, global_step)
            self.writer.add_scalar('Train/Loss_Recon', recon_val, global_step)
            self.writer.add_scalar('Train/Loss_Eikonal', loss_eik.item(), global_step)

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def validate(self, epoch):
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for roi, coords, gt_sdf in self.val_loader:
                roi = roi.to(self.device)
                coords = coords.to(self.device)
                gt_sdf = gt_sdf.to(self.device)

                pred_sdf = self.model(roi, coords)
                loss = sdf_loss(pred_sdf, gt_sdf)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(self.val_loader)

        self.writer.add_scalar('Val/Loss', avg_val_loss, epoch)
        self.scheduler.step(avg_val_loss)

        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('Train/LR', current_lr, epoch)

        return avg_val_loss

    def save_checkpoint(self, epoch, is_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.cfg
        }

        last_path = os.path.join(self.ckpt_dir, "last.pth")
        torch.save(state, last_path)

        if is_best:
            best_path = os.path.join(self.ckpt_dir, "best_model.pth")
            torch.save(state, best_path)