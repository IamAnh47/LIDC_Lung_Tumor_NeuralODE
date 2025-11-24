import os
import yaml
import json
import numpy as np
import trimesh
import pylidc as pl
from tqdm import tqdm
from pylidc.utils import consensus

# Import các hàm xử lý từ src (Tái sử dụng code để đảm bảo nhất quán)
from src.data.dicom_loader import DicomLoader
from src.data.preprocessing import resample_volume, crop_roi
from src.data.generation import mesh_from_mask


def main():
    print("🚀 BẮT ĐẦU KIỂM TRA DỮ LIỆU ĐÃ XỬ LÝ (TEST SET)...")

    # 1. Load Config
    if not os.path.exists("configs/config.yaml"):
        print("❌ Không tìm thấy configs/config.yaml")
        return

    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Đường dẫn
    RAW_DIR = os.path.abspath(cfg['paths']['raw_data'])
    PROCESSED_DIR = os.path.abspath(cfg['paths']['processed_data'])
    OUTPUT_CHECK_DIR = "test_prepare"  # Thư mục yêu cầu

    # Tham số (Phải giống hệt lúc prepare_data)
    ROI_SIZE = tuple(cfg['data']['roi_size'])
    TARGET_SPACING = tuple(cfg['data']['target_spacing'])

    # Tạo thư mục output
    os.makedirs(OUTPUT_CHECK_DIR, exist_ok=True)
    print(f"📂 Kết quả sẽ lưu tại: {os.path.abspath(OUTPUT_CHECK_DIR)}")

    # 2. Load danh sách tập Test
    split_path = os.path.join(PROCESSED_DIR, "split_data.json")
    if not os.path.exists(split_path):
        print("❌ Chưa có file split_data.json. Hãy chạy prepare_data.py trước!")
        return

    with open(split_path, "r") as f:
        splits = json.load(f)
        test_ids = splits["test"]  # Chỉ lấy tập test để check

    print(f"📦 Tìm thấy {len(test_ids)} mẫu trong tập Test.")

    # 3. Khởi tạo Loader
    loader = DicomLoader(RAW_DIR)

    # 4. Vòng lặp xử lý (Re-generate Mesh)
    for file_id in tqdm(test_ids, desc="Generating GT Meshes"):
        try:
            # file_id dạng: "LIDC-IDRI-0074_nodule0"
            # Cần tách ra PID và Nodule Index
            parts = file_id.split('_nodule')
            pid = parts[0]
            nodule_idx = int(parts[1])

            # Load lại dữ liệu gốc
            vol_orig, spacing_orig, nodules = loader.load_patient_data(pid)

            if vol_orig is None or nodule_idx >= len(nodules):
                print(f"⚠️ Không tìm thấy dữ liệu gốc cho {file_id}")
                continue

            # Lấy đúng nodule group
            annots = nodules[nodule_idx]

            # --- TÁI TẠO MESH (Quy trình y hệt prepare_data) ---

            # A. Consensus
            mask_orig, cbbox, _ = consensus(annots, clevel=0.5, pad=10)

            # Crop Volume gốc (để lấy context nếu cần, ở đây chỉ cần mask để tạo mesh)
            # Nhưng resample cần cả 2 để đồng bộ
            vol_nodule = vol_orig[cbbox]

            # B. Resample
            _, mask_res = resample_volume(vol_nodule, mask_orig, spacing_orig, TARGET_SPACING)

            # C. Crop ROI (Để đúng kích thước 64x64x32)
            # Lưu ý: Ta cần crop mask theo đúng logic đã làm với ảnh
            # Để đơn giản, ta truyền dummy volume vào hàm crop_roi
            dummy_vol = np.zeros_like(mask_res, dtype=np.float32)
            _, roi_mask = crop_roi(dummy_vol, mask_res, size=ROI_SIZE)

            if roi_mask is None:
                print(f"⚠️ Mask rỗng sau khi crop: {file_id}")
                continue

            # D. Tạo Mesh (Ground Truth)
            # Đây là hàm dùng Marching Cubes


            #[Image of marching cubes algorithm]

            mesh = mesh_from_mask(roi_mask, spacing=TARGET_SPACING)

            if mesh:
                # E. Export
                # Căn giữa để dễ xem trên Blender
                mesh.apply_translation(-mesh.centroid)

                save_name = f"{file_id}_GT.obj"
                mesh.export(os.path.join(OUTPUT_CHECK_DIR, save_name))

                # (Tùy chọn) Lưu thêm file Point Cloud từ .npz để so sánh
                # copy_point_cloud(PROCESSED_DIR, file_id, OUTPUT_CHECK_DIR)
            else:
                print(f"⚠️ Không tạo được Mesh cho {file_id}")

        except Exception as e:
                print(f"❌ Lỗi khi xử lý {file_id}: {e}")
    print("\n✅ HOÀN TẤT! Hãy mở thư mục 'test_prepare' và kéo file .obj vào Blender.")


def copy_point_cloud(processed_dir, file_id, output_dir):
    """
    Hàm phụ: Trích xuất điểm từ file .npz ra file .obj (dạng đám mây điểm)
    để xem model thực sự 'nhìn thấy' những điểm nào.
    """
    try:
        npz_path = os.path.join(processed_dir, "sdfs", f"{file_id}.npz")
        data = np.load(npz_path)
        points = data['points']
        sdfs = data['sdfs']  # Hoặc 'values' tùy tên đặt

        # Lọc lấy các điểm bề mặt (SDF gần 0) để visualize cho nhẹ
        # Hoặc xuất hết
        pcd = trimesh.points.PointCloud(points)

        # Căn giữa (Cần khớp với mesh ở trên nếu muốn chồng hình)
        # pcd.apply_translation(-pcd.centroid)

        pcd.export(os.path.join(output_dir, f"{file_id}_POINTS.obj"))
    except:
        pass


if __name__ == "__main__":
    main()