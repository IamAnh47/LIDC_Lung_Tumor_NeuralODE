import numpy as np
import pylidc as pl
import configparser
import os
import glob
import pydicom  # Cần thêm thư viện này để đọc header

# --- VÁ LỖI NUMPY ---
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'bool'):
    np.bool = bool


# ---------------------

class DicomLoader:
    def __init__(self, raw_data_dir):
        self.raw_data_dir = raw_data_dir
        self.path_cache = {}  # Cache đường dẫn để chạy nhanh hơn

        if not hasattr(configparser, 'SafeConfigParser'):
            configparser.SafeConfigParser = configparser.ConfigParser

        self._activate_smart_path_finding()

    def _activate_smart_path_finding(self):
        """
        Hack nâng cao: Tìm đúng folder chứa SeriesInstanceUID cụ thể.
        """
        root_dir = self.raw_data_dir
        cache = self.path_cache

        def smart_path_method(scan_instance):
            current_id = scan_instance.patient_id
            target_series_uid = scan_instance.series_instance_uid

            # 1. Kiểm tra Cache trước (đỡ phải quét lại ổ cứng)
            cache_key = f"{current_id}_{target_series_uid}"
            if cache_key in cache:
                return cache[cache_key]

            # 2. Tìm tất cả các file dcm của bệnh nhân này
            # (Giả định cấu trúc: raw/LIDC-IDRI-xxxx/...)
            patient_dir = os.path.join(root_dir, current_id)
            if not os.path.exists(patient_dir):
                # Fallback: Quét rộng hơn nếu cấu trúc folder lạ
                patient_dir = root_dir

                # Duyệt cây thư mục để tìm đúng Series
            # Đây là bước quan trọng: Một bệnh nhân có thể có nhiều folder con
            for root, dirs, files in os.walk(patient_dir):
                dcm_files = [f for f in files if f.endswith('.dcm')]
                if not dcm_files:
                    continue

                # Đọc thử file đầu tiên trong folder này để xem UID
                try:
                    first_file = os.path.join(root, dcm_files[0])
                    # stop_before_pixels=True giúp đọc cực nhanh (chỉ lấy header)
                    ds = pydicom.dcmread(first_file, stop_before_pixels=True)

                    if ds.SeriesInstanceUID == target_series_uid:
                        # Tìm thấy đúng chuồng! Lưu cache và trả về
                        cache[cache_key] = root
                        return root
                except Exception:
                    continue

            # Nếu quét hết mà không thấy (hoặc file lỗi)
            # Thử phương án cũ (May rủi): Trả về folder đầu tiên tìm thấy có tên ID
            # (Để tránh crash hoàn toàn, dù có thể sai series)
            search_pattern = os.path.join(root_dir, "**", current_id, "**", "*.dcm")
            found_files = glob.glob(search_pattern, recursive=True)
            if found_files:
                return os.path.dirname(found_files[0])

            raise FileNotFoundError(f"❌ Không tìm thấy Series {target_series_uid} của {current_id}")

        pl.Scan.get_path_to_dicom_files = smart_path_method

    def get_all_patient_ids(self):
        # Quét folder cấp 1 để lấy ID nhanh
        search_pattern = os.path.join(self.raw_data_dir, "LIDC-IDRI-*")
        folders = glob.glob(search_pattern)
        ids = [os.path.basename(f) for f in folders]
        return sorted(ids)

    def load_patient_data(self, patient_id):
        try:
            # Lấy bản ghi đầu tiên (hoặc loop qua các scan nếu muốn)
            # Ở đây ta lấy scan đầu tiên tìm được trong DB
            scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).first()
        except Exception as e:
            print(f"   ⚠️ Lỗi query DB cho {patient_id}: {e}")
            return None, None, None

        if scan is None:
            return None, None, None

        try:
            # Load list các file DICOM
            images = scan.load_all_dicom_images(verbose=False)

            # --- FIX LỖI CRASH: Kiểm tra list rỗng ---
            if not images or len(images) == 0:
                print(
                    f"   ⚠️ Cảnh báo: Tìm thấy Scan nhưng không đọc được ảnh nào (Folder rỗng hoặc sai UID). Bỏ qua {patient_id}.")
                return None, None, None

            # Stack thành khối 3D và chuyển sang HU
            vol = np.stack([s.pixel_array * s.RescaleSlope + s.RescaleIntercept for s in images])
            vol = vol.astype(np.float32)

        except Exception as e:
            print(f"   ⚠️ Lỗi đọc dữ liệu {patient_id}: {e}")
            return None, None, None

        spacing = (scan.slice_spacing, scan.pixel_spacing, scan.pixel_spacing)
        nodules = scan.cluster_annotations()

        return vol, spacing, nodules