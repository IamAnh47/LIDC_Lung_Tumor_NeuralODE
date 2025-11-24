import matplotlib.pyplot as plt
import numpy as np
import os

class Visualizer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_loss(self, train_losses, val_losses, filename="loss_curve.png"):
        """
        Vẽ biểu đồ Loss theo epoch.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss', color='blue')
        # Val loss thường ít điểm hơn (do val_every), cần x-axis tương ứng
        # Giả sử val_losses là list các tuple (epoch, loss) hoặc list loss
        if len(val_losses) > 0:
            if isinstance(val_losses[0], tuple):
                val_x, val_y = zip(*val_losses)
                plt.plot(val_x, val_y, label='Val Loss', color='red', marker='o')
            else:
                plt.plot(val_losses, label='Val Loss', color='red')
                
        plt.title('Training Progress (Neural ODE)')
        plt.xlabel('Epoch')
        plt.ylabel('SDF Loss')
        plt.legend()
        plt.grid(True)
        
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"📈 Đã lưu biểu đồ Loss tại: {save_path}")

    def visualize_slices(self, volume, mask=None, filename="sample_slices.png"):
        """
        Hiển thị 3 lát cắt trực giao (Axial, Coronal, Sagittal) của khối 3D.
        Giúp kiểm tra xem ROI crop có đúng khối u không.
        """
        # Lấy index trung tâm
        z, y, x = np.array(volume.shape) // 2
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Lát cắt Axial (Z)
        axes[0].imshow(volume[z, :, :], cmap='gray')
        if mask is not None:
            axes[0].contour(mask[z, :, :], colors='red', linewidths=0.5)
        axes[0].set_title(f"Axial (Z={z})")
        
        # Lát cắt Coronal (Y)
        axes[1].imshow(volume[:, y, :], cmap='gray')
        if mask is not None:
            axes[1].contour(mask[:, y, :], colors='red', linewidths=0.5)
        axes[1].set_title(f"Coronal (Y={y})")
        
        # Lát cắt Sagittal (X)
        axes[2].imshow(volume[:, :, x], cmap='gray')
        if mask is not None:
            axes[2].contour(mask[:, :, x], colors='red', linewidths=0.5)
        axes[2].set_title(f"Sagittal (X={x})")
        
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"🖼️ Đã lưu ảnh lát cắt tại: {save_path}")

# --- Code test nhanh ---
if __name__ == "__main__":
    # Test vẽ biểu đồ giả
    vis = Visualizer("outputs/test_vis")
    t_loss = [0.5, 0.4, 0.3, 0.25, 0.2]
    v_loss = [(1, 0.45), (3, 0.35), (5, 0.22)]
    vis.plot_loss(t_loss, v_loss)