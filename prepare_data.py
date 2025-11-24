import os
import yaml
import numpy as np
from tqdm import tqdm
import json
import random
import pylidc as pl  # <--- THÊM DÒNG NÀY

# Import các module tự viết
from src.data.dicom_loader import DicomLoader
from src.data.preprocessing import normalize_hu, resample_volume, crop_roi
from src.data.generation import mesh_from_mask, generate_sdf_points
from pylidc.utils import consensus


def main():
    # 1. Load Config
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Lấy đường dẫn tuyệt đối
    RAW_DIR = os.path.abspath(cfg['paths']['raw_data'])
    PROCESSED_DIR = os.path.abspath(cfg['paths']['processed_data'])

    # Tham số xử lý
    ROI_SIZE = tuple(cfg['data']['roi_size'])
    TARGET_SPACING = tuple(cfg['data']['target_spacing'])
    NUM_SAMPLES = cfg['data']['sdf_samples']

    # Tạo thư mục output
    os.makedirs(os.path.join(PROCESSED_DIR, "rois"), exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_DIR, "sdfs"), exist_ok=True)

    # 2. Khởi tạo DicomLoader
    loader = DicomLoader(RAW_DIR)

    # Lấy danh sách bệnh nhân
    patient_ids = loader.get_all_patient_ids()
    print(f"🚀 Tìm thấy {len(patient_ids)} bệnh nhân. Bắt đầu xử lý...")

    processed_records = []

    # Biến đếm thống kê
    stats = {
        "no_nodules": 0,
        "consensus_empty": 0,
        "too_complex": 0,
        "too_small": 0,
        "success": 0
    }

    for pid in tqdm(patient_ids):
        try:
            # Query để check xem có nodule không trước khi load ảnh nặng
            # (Dùng loader.load_patient_data đã bao gồm bước này, nhưng tách ra để đếm stats chuẩn hơn)
            scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
            if not scan: continue

            nodules = scan.cluster_annotations()
            if not nodules:
                stats["no_nodules"] += 1
                continue

            # Dùng Loader để lấy dữ liệu thô (Ảnh + Spacing)
            # Lưu ý: Hàm load_patient_data trong dicom_loader.py trả về 3 giá trị
            vol_orig, spacing_orig, nodules = loader.load_patient_data(pid)

            if vol_orig is None:
                continue

            # Xử lý từng khối u
            for i, annots in enumerate(nodules):
                # --- LỌC LỖI 1: QUÁ PHỨC TẠP ---
                if len(annots) > 4:
                    # print(f"   ⚠️ Bỏ qua Nodule {i} của {pid}: >4 anns.")
                    stats["too_complex"] += 1
                    continue
                # ==============================================================================
                # [THAM KHẢO SAU NÀY] BỘ LỌC ĐỘ ÁC TÍNH (MALIGNANCY FILTER)
                # ------------------------------------------------------------------------------
                # Mỗi annotation có thuộc tính .malignancy (1: Lành tính -> 5: Ác tính)
                # Ta tính trung bình cộng đánh giá của các bác sĩ.
                #
                # avg_malignancy = np.mean([a.malignancy for a in annots])
                #
                # if avg_malignancy < 3:
                #     # print(f"   ⏩ Bỏ qua Nodule {i}: Khả năng cao là lành tính (Score: {avg_malignancy:.1f})")
                #     continue
                # ==============================================================================
                # 1. Consensus Mask
                try:
                    mask_orig, cbbox, _ = consensus(annots, clevel=0.5, pad=10)
                except Exception:
                    stats["consensus_empty"] += 1
                    continue

                # --- LỌC LỖI 2: MASK QUÁ BÉ ---
                if np.sum(mask_orig) < 50:
                    stats["consensus_empty"] += 1
                    continue

                # Crop volume gốc theo bbox của mask
                # cbbox là tuple của các slice objects
                vol_nodule = vol_orig[cbbox]

                # 2. Resample & Normalize
                vol_res, mask_res = resample_volume(vol_nodule, mask_orig, spacing_orig, TARGET_SPACING)
                vol_norm = normalize_hu(vol_res)

                # 3. Crop Fixed ROI
                roi_vol, roi_mask = crop_roi(vol_norm, mask_res, size=ROI_SIZE)

                if roi_vol is None: continue

                # 4. Generate Mesh & Check Size
                mesh = mesh_from_mask(roi_mask, spacing=TARGET_SPACING)

                # --- LỌC LỖI 3: MESH LỖI ---
                if mesh is None or len(mesh.vertices) < 10:
                    stats["too_small"] += 1
                    continue

                # 5. Generate SDF Data (Dùng lại mesh đã tạo)
                points, sdfs = generate_sdf_points(mesh, num_samples=NUM_SAMPLES, roi_size=ROI_SIZE)

                if points is None: continue

                # 6. Save Disk
                file_id = f"{pid}_nodule{i}"

                np.save(os.path.join(PROCESSED_DIR, "rois", f"{file_id}.npy"), roi_vol)
                np.savez(os.path.join(PROCESSED_DIR, "sdfs", f"{file_id}.npz"), points=points, sdfs=sdfs)

                stats["success"] += 1
                processed_records.append(file_id)

        except Exception as e:
            print(f"⚠️ Lỗi {pid}: {e}")
            continue

    print("\n📊 BÁO CÁO CHI TIẾT:")
    print(f"❌ Không có nodule: {stats['no_nodules']}")
    print(f"❌ Mask rỗng/bé: {stats['consensus_empty']}")
    print(f"❌ Quá phức tạp: {stats['too_complex']}")
    print(f"❌ Mesh lỗi: {stats['too_small']}")
    print(f"✅ THÀNH CÔNG: {stats['success']} mẫu")

    # 4. Chia tập Train/Val/Test
    if len(processed_records) > 0:
        print(f"\n📦 Tổng cộng: {len(processed_records)} mẫu sạch.")
        print("✂️ Đang chia tập dữ liệu...")

        random.seed(42)
        random.shuffle(processed_records)

        n_total = len(processed_records)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.1)

        split_dict = {
            "train": processed_records[:n_train],
            "val": processed_records[n_train:n_train + n_val],
            "test": processed_records[n_train + n_val:]
        }

        with open(os.path.join(PROCESSED_DIR, "split_data.json"), "w") as f:
            json.dump(split_dict, f, indent=4)

        print(
            f"✅ Đã lưu split_data.json (Train: {len(split_dict['train'])}, Val: {len(split_dict['val'])}, Test: {len(split_dict['test'])})")
    else:
        print("❌ Không có dữ liệu nào được xử lý thành công.")


if __name__ == "__main__":
    main()