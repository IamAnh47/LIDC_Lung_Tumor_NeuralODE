import numpy as np
import trimesh
from skimage.measure import marching_cubes

def generate_mesh_from_sdf(sdf_volume, spacing=(1.0, 1.0, 1.0), level=0.0):
    """
    Chạy thuật toán Marching Cubes trên khối Volume SDF.
    - sdf_volume: Khối 3D chứa giá trị SDF dự đoán.
    - level: Mức trích xuất (SDF=0 là bề mặt).
    """
    try:
        # Thuật toán Marching Cubes
        verts, faces, normals, _ = marching_cubes(sdf_volume, level=level, spacing=spacing)
        
        # Tạo đối tượng Mesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        
        return mesh
    except ValueError:
        # Trường hợp không tìm thấy bề mặt (SDF toàn dương hoặc toàn âm)
        return None
    except Exception as e:
        print(f"⚠️ Lỗi Marching Cubes: {e}")
        return None