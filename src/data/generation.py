import numpy as np
import trimesh
from skimage.measure import marching_cubes


def mesh_from_mask(mask, spacing=(1.0, 1.0, 1.0)):
    try:
        # level=0.5 vì mask là nhị phân 0/1
        verts, faces, normals, _ = marching_cubes(mask, level=0.5, spacing=spacing)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        return mesh
    except Exception:
        return None


def generate_sdf_points(mesh, num_samples=10000, roi_size=(64, 64, 32)):
    """
    Sinh dữ liệu SDF với chiến lược 'Siết chặt bề mặt' (Tight Surface Sampling).
    Mục tiêu: Ép mô hình học biên dạng sắc nét, tránh bị nhiễu bởi vùng đệm quá dày.
    """
    if mesh is None: return None, None

    # Quay về tỷ lệ vàng: 50% Surface - 50% Uniform
    # Hoặc 60% Surface - 40% Uniform

    n_surf = int(num_samples * 0.5)  # Giảm từ 0.8 xuống 0.5
    points_surf, _ = trimesh.sample.sample_surface(mesh, n_surf)
    points_surf += np.random.normal(0, 1.0, points_surf.shape)  # Giữ nhiễu 1.0mm để tạo dốc

    n_uniform = num_samples - n_surf  # Tăng lên 50% để model học được nền tốt hơn

    # Random trong kích thước ROI (Z, Y, X)
    z_rand = np.random.uniform(0, roi_size[0], n_uniform)
    y_rand = np.random.uniform(0, roi_size[1], n_uniform)
    x_rand = np.random.uniform(0, roi_size[2], n_uniform)

    # Stack lại (Thứ tự Z, Y, X khớp với roi_size)
    points_uniform = np.stack([z_rand, y_rand, x_rand], axis=1)

    # Gộp tất cả lại
    all_points = np.vstack([points_surf, points_uniform])

    # Kiểm tra biên (Clip points nằm ngoài ROI để tránh lỗi Index out of bound sau này)
    all_points[:, 0] = np.clip(all_points[:, 0], 0, roi_size[0] - 1)
    all_points[:, 1] = np.clip(all_points[:, 1], 0, roi_size[1] - 1)
    all_points[:, 2] = np.clip(all_points[:, 2], 0, roi_size[2] - 1)

    # --- TÍNH SDF ---
    # Trimesh mặc định: Dương = Ngoài, Âm = Trong (đôi khi ngược lại tùy version)
    # DeepSDF Standard: Trong (-), Ngoài (+)
    sdf_values = trimesh.proximity.signed_distance(mesh, all_points)

    # ĐẢO DẤU ĐỂ ĐÚNG QUY ƯỚC (Trong Âm, Ngoài Dương)
    # (Nếu check debug thấy tâm u dương thì bỏ dòng này, nhưng thường là cần đảo)
    sdf_values = -sdf_values

    return all_points, sdf_values