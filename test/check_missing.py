import os
import glob

# Đường dẫn dữ liệu
raw_dir = r"/data/raw/LIDC-IDRI"

# 1. Lấy danh sách thực tế đang có
folders = [os.path.basename(p) for p in glob.glob(os.path.join(raw_dir, "LIDC-IDRI-*"))]
existing_ids = set(folders)

print(f"📂 Thực tế tìm thấy: {len(existing_ids)} folder.")

# 2. Tạo danh sách kỳ vọng (0001 -> 0450)
expected_ids = {f"LIDC-IDRI-{i:04d}" for i in range(1, 451)}

# 3. Tìm kẻ mất tích (Hiệu của 2 tập hợp)
missing = sorted(list(expected_ids - existing_ids))

print("-" * 30)
if missing:
    print(f"❌ PHÁT HIỆN {len(missing)} BỆNH NHÂN BỊ THIẾU:")
    for m in missing:
        print(f"   - {m}")
else:
    print("✅ Đủ cả! Không thiếu ai (Có thể do logic đếm file bị nhầm đâu đó).")

# 4. Kiểm tra xem có ông nào "lạ" không (Ngoài vùng 1-450)
extras = sorted(list(existing_ids - expected_ids))
if extras:
    print("\n⚠️ CÁC FOLDER LẠ (Nằm ngoài dải 0001-0450):")
    for e in extras:
        print(f"   - {e}")