import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from .losses import sdf_loss, eikonal_loss

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config
        self.device = device
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=float(config['train']['learning_rate'])
        )
        
        # Scheduler (Giảm LR khi loss không giảm nữa)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Mixed Precision Scaler (Tăng tốc độ train)
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Logging
        # self.writer = SummaryWriter(log_dir=os.path.join(config['paths']['experiment_dir'],'exp_01_nnunet', "logs"))
        # self.ckpt_dir = os.path.join(config['paths']['experiment_dir'],'exp_01_nnunet', "checkpoints")
        self.writer = SummaryWriter(log_dir=os.path.join(config['paths']['experiment_dir'],'exp_02_resnet', "logs"))
        self.ckpt_dir = os.path.join(config['paths']['experiment_dir'],'exp_02_resnet', "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (roi, coords, gt_sdf) in enumerate(pbar):
            roi = roi.to(self.device)
            coords = coords.to(self.device)
            gt_sdf = gt_sdf.to(self.device)
            
            # Bật tính năng theo dõi Gradient cho coords (để tính Eikonal Loss)
            coords.requires_grad_(True)
            
            # --- FORWARD PASS (Mixed Precision) ---
            with torch.cuda.amp.autocast():
                pred_sdf = self.model(roi, coords)
                
                # Tính Loss
                loss_recon = sdf_loss(pred_sdf, gt_sdf)
                
                # Eikonal Loss
                loss_eik = eikonal_loss(pred_sdf, coords)
                
                loss = loss_recon + 0.001 * loss_eik
            
            # --- BACKWARD PASS ---
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
            # Update Progress Bar
            pbar.set_postfix({'Loss': loss.item(), 'Recon': loss_recon.item()})
            
            # Log step
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/Loss', loss.item(), global_step)

        return total_loss / len(self.train_loader)

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
        
        print(f"Evaluate Epoch {epoch}: Val Loss = {avg_val_loss:.6f}")
        return avg_val_loss

    def save_checkpoint(self, epoch, is_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        # Lưu last.pth
        torch.save(state, os.path.join(self.ckpt_dir, "last.pth"))
        # Lưu best.pth
        if is_best:
            torch.save(state, os.path.join(self.ckpt_dir, "best_model.pth"))
            print(f"🎉 Đã lưu Best Model tại epoch {epoch}!")