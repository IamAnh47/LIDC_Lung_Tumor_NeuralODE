import numpy as np
import trimesh
from scipy.spatial import cKDTree

def compute_chamfer_distance(pred_points, gt_points):
    """
    Tính khoảng cách Chamfer trung bình giữa 2 đám mây điểm.
    Chamfer Distance = mean(dist(pred, gt)) + mean(dist(gt, pred))
    """
    # Xây dựng cây tìm kiếm
    tree_gt = cKDTree(gt_points)
    dist_pred_to_gt, _ = tree_gt.query(pred_points)
    
    tree_pred = cKDTree(pred_points)
    dist_gt_to_pred, _ = tree_pred.query(gt_points)
    
    chamfer_dist = np.mean(dist_pred_to_gt) + np.mean(dist_gt_to_pred)
    return chamfer_dist

def compute_hausdorff_distance(pred_points, gt_points, percentile=95):
    """
    Tính khoảng cách Hausdorff (95%) để loại bỏ nhiễu ngoại lai.
    """
    tree_gt = cKDTree(gt_points)
    dist_pred_to_gt, _ = tree_gt.query(pred_points)
    
    tree_pred = cKDTree(pred_points)
    dist_gt_to_pred, _ = tree_pred.query(gt_points)
    
    hd_pred = np.percentile(dist_pred_to_gt, percentile)
    hd_gt = np.percentile(dist_gt_to_pred, percentile)
    
    return max(hd_pred, hd_gt)

def compute_all_metrics(pred_mesh, gt_mesh, num_samples=30000):
    """
    Hàm tổng hợp để tính tất cả các metric cùng lúc.
    Input: 2 đối tượng trimesh.
    Output: Dictionary chứa kết quả.
    """
    if pred_mesh is None or gt_mesh is None:
        return {"Chamfer": np.nan, "Hausdorff95": np.nan, "ASSD": np.nan}

    # 1. Lấy mẫu điểm trên bề mặt (Surface Sampling)
    # Đây là cách chuẩn nhất để so sánh 2 mesh có topology khác nhau
    try:
        pred_points, _ = trimesh.sample.sample_surface(pred_mesh, num_samples)
        gt_points, _ = trimesh.sample.sample_surface(gt_mesh, num_samples)
    except Exception as e:
        print(f"⚠️ Lỗi sampling mesh: {e}")
        return {"Chamfer": np.nan, "Hausdorff95": np.nan}

    # 2. Tính toán
    chamfer = compute_chamfer_distance(pred_points, gt_points)
    hd95 = compute_hausdorff_distance(pred_points, gt_points, percentile=95)
    
    # ASSD (Average Symmetric Surface Distance) - Giống Chamfer nhưng không bình phương
    # Ở trên hàm chamfer trả về mean khoảng cách (L1) nên chính là ASSD * 2
    # Ta tính lại cho rõ ràng:
    tree_gt = cKDTree(gt_points)
    d1, _ = tree_gt.query(pred_points)
    tree_pred = cKDTree(pred_points)
    d2, _ = tree_pred.query(gt_points)
    assd = (np.mean(d1) + np.mean(d2)) / 2.0

    return {
        "Chamfer": chamfer,       # mm (càng thấp càng tốt)
        "Hausdorff95": hd95,      # mm (càng thấp càng tốt)
        "ASSD": assd              # mm (càng thấp càng tốt)
    }