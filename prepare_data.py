import os
import yaml
import numpy as np
from tqdm import tqdm
import json
import random
import pylidc as pl
import glob

# Import các module tự viết
from src.data.dicom_loader import DicomLoader
from src.data.preprocessing import normalize_hu, resample_volume, crop_roi
from src.data.generation import mesh_from_mask, generate_sdf_points
from pylidc.utils import consensus


def main():
    # 1. Load Config
    config_path = "configs/config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Không tìm thấy file config tại: {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Setup đường dẫn
    RAW_DIR = os.path.abspath(cfg['paths']['raw_data'])
    PROCESSED_DIR = os.path.abspath(cfg['paths']['processed_data'])
    ROI_DIR = os.path.join(PROCESSED_DIR, "rois")
    SDF_DIR = os.path.join(PROCESSED_DIR, "sdfs")

    os.makedirs(ROI_DIR, exist_ok=True)
    os.makedirs(SDF_DIR, exist_ok=True)

    ROI_SIZE = tuple(cfg['data']['roi_size'])
    TARGET_SPACING = tuple(cfg['data']['target_spacing'])
    NUM_SAMPLES = cfg['data']['sdf_samples']

    # ==========================================================================
    # 🛠️ QUẢN LÝ TIẾN ĐỘ (SỔ ĐIỂM DANH)
    # ==========================================================================
    LOG_FILE = os.path.join(PROCESSED_DIR, "processed_patients.json")

    processed_pids = set()
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                processed_pids = set(json.load(f))
        except:
            pass

    # 2. Init Loader & Filter
    loader = DicomLoader(RAW_DIR)
    all_patients = loader.get_all_patient_ids()

    # Chỉ chạy những người chưa có trong sổ điểm danh
    target_patients = [p for p in all_patients if p not in processed_pids]
    target_patients.sort()

    if not target_patients:
        print("🎉 Không có dữ liệu mới. Chuyển sang bước tổng hợp.")
    else:
        print(f"🚀 Tìm thấy {len(target_patients)} bệnh nhân MỚI. Bắt đầu xử lý...")

    # --- BIẾN THỐNG KÊ CHO ĐỢT CHẠY NÀY ---
    stats = {
        "no_nodules": 0,  # Bệnh nhân không có nodule
        "consensus_empty": 0,  # Mask rỗng/bé
        "too_complex": 0,  # > 4 bác sĩ
        "too_small": 0,  # Mesh lỗi
        "success": 0  # Số khối u thành công
    }

    batch_counter = 0

    # 3. Vòng lặp chính (Xử lý người mới)
    if target_patients:
        for pid in tqdm(target_patients, desc="Processing"):
            try:
                # Query & Check
                scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
                if not scan:
                    completed_patients = list(processed_pids)  # Đánh dấu xong để lần sau ko check nữa
                    continue

                nodules = scan.cluster_annotations()
                if not nodules:
                    stats["no_nodules"] += 1
                    processed_pids.add(pid)  # Đánh dấu xong
                    continue

                # Load Data
                vol_orig, spacing_orig, nodules = loader.load_patient_data(pid)
                if vol_orig is None:
                    continue  # Lỗi load ảnh, ko đánh dấu xong để lần sau thử lại

                # Xử lý từng khối u
                for i, annots in enumerate(nodules):
                    # --- LỌC LỖI 1 ---
                    if len(annots) > 4:
                        stats["too_complex"] += 1
                        continue

                    # --- LỌC LỖI 2 ---
                    try:
                        mask_orig, cbbox, _ = consensus(annots, clevel=0.5, pad=10)
                    except:
                        stats["consensus_empty"] += 1
                        continue

                    if np.sum(mask_orig) < 50:
                        stats["consensus_empty"] += 1
                        continue

                    # Preprocessing
                    vol_nodule = vol_orig[cbbox]
                    vol_res, mask_res = resample_volume(vol_nodule, mask_orig, spacing_orig, TARGET_SPACING)
                    vol_norm = normalize_hu(vol_res)
                    roi_vol, roi_mask = crop_roi(vol_norm, mask_res, size=ROI_SIZE)

                    if roi_vol is None: continue

                    # --- LỌC LỖI 3 ---
                    mesh = mesh_from_mask(roi_mask, spacing=TARGET_SPACING)
                    if mesh is None or len(mesh.vertices) < 10:
                        stats["too_small"] += 1
                        continue

                    # Generate Data
                    points, sdfs = generate_sdf_points(mesh, num_samples=NUM_SAMPLES, roi_size=ROI_SIZE)
                    if points is None: continue

                    # Save
                    file_id = f"{pid}_nodule{i}"
                    np.save(os.path.join(ROI_DIR, f"{file_id}.npy"), roi_vol)
                    np.savez(os.path.join(SDF_DIR, f"{file_id}.npz"), points=points, sdfs=sdfs)

                    stats["success"] += 1

                # Xong bệnh nhân này -> Ghi vào sổ điểm danh
                processed_pids.add(pid)
                batch_counter += 1

                # Lưu log định kỳ (10 người/lần)
                if batch_counter % 10 == 0:
                    with open(LOG_FILE, "w", encoding="utf-8") as f:
                        json.dump(sorted(list(processed_pids)), f, indent=4)

            except Exception as e:
                # print(f"⚠️ Lỗi {pid}: {e}")
                continue

        # Lưu log lần cuối
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(sorted(list(processed_pids)), f, indent=4)

        # --- IN BÁO CÁO CHI TIẾT (Chỉ cho đợt chạy này) ---
        print("\n📊 BÁO CÁO CHI TIẾT (Dữ liệu mới xử lý):")
        print(f"❌ Không có nodule: {stats['no_nodules']}")
        print(f"❌ Mask rỗng/bé: {stats['consensus_empty']}")
        print(f"❌ Quá phức tạp: {stats['too_complex']}")
        print(f"❌ Mesh lỗi: {stats['too_small']}")
        print(f"✅ THÀNH CÔNG: {stats['success']} mẫu")

    else:
        print("\n(Không có dữ liệu mới để báo cáo chi tiết)")

    # ==========================================================================
    # 4. TỔNG HỢP & CHIA TẬP (TOÀN BỘ DỮ LIỆU CŨ + MỚI)
    # ==========================================================================
    # Quét ổ cứng để lấy tổng số thực tế
    all_npy_files = glob.glob(os.path.join(ROI_DIR, "*.npy"))

    if not all_npy_files:
        print("❌ Không tìm thấy dữ liệu nào trong thư mục processed.")
        return

    total_samples = len(all_npy_files)
    print(f"\n📦 Tổng cộng: {total_samples} mẫu sạch (Cũ + Mới).")
    print("✂️ Đang chia tập dữ liệu...")

    valid_records = [os.path.basename(f).replace(".npy", "") for f in all_npy_files]
    valid_records.sort()

    random.seed(42)
    random.shuffle(valid_records)

    n_train = int(total_samples * 0.7)
    n_val = int(total_samples * 0.1)

    split_dict = {
        "train": valid_records[:n_train],
        "val": valid_records[n_train:n_train + n_val],
        "test": valid_records[n_train + n_val:]
    }

    with open(os.path.join(PROCESSED_DIR, "split_data.json"), "w", encoding="utf-8") as f:
        json.dump(split_dict, f, indent=4)

    print(
        f"✅ Đã lưu split_data.json (Train: {len(split_dict['train'])}, Val: {len(split_dict['val'])}, Test: {len(split_dict['test'])})")


if __name__ == "__main__":
    main()