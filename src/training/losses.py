import torch
import torch.nn.functional as F


def sdf_loss(pred, gt):
    """
    SDF Loss sửa đổi: CHỈ KẸP GROUND TRUTH, KHÔNG KẸP PREDICTION.
    Để Gradient luôn chảy về mô hình.
    """
    # 1. CẤU HÌNH KẸP (Clamping)
    CLAMP_VAL = 0.5

    # --- SỬA TẠI ĐÂY: Bỏ clamp cho pred ---
    # pred_clamped = torch.clamp(pred, -CLAMP_VAL, CLAMP_VAL) <--- XÓA DÒNG NÀY

    # Chỉ kẹp Ground Truth (Đáp án)
    # Ý nghĩa: Nếu điểm thật cách xa quá 10mm (0.5), ta coi như nó chỉ cách 10mm thôi.
    # Điều này giúp model không bị nhiễu bởi các điểm xa lắc xa lơ.
    gt_clamped = torch.clamp(gt, -CLAMP_VAL, CLAMP_VAL)

    # Tính L1 Error: So sánh Pred (tự do) với GT (đã kẹp)
    error = torch.abs(pred - gt_clamped)

    # 2. TRỌNG SỐ (Weighting)
    # Giữ nguyên phạt nặng vùng bên trong
    weights = torch.ones_like(gt)
    weights[gt < 0] = 15.0

    return (error * weights).mean()


def eikonal_loss(model_output, coords):
    """
    Giữ nguyên Eikonal Loss để đảm bảo tính chất hình học.
    """
    gradients = torch.autograd.grad(
        outputs=model_output,
        inputs=coords,
        grad_outputs=torch.ones_like(model_output),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    grad_norm = gradients.norm(2, dim=-1)
    return ((grad_norm - 1.0) ** 2).mean()