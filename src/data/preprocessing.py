import numpy as np
from scipy.ndimage import zoom

def normalize_hu(vol, min_hu=-1000, max_hu=400):
    """
    Chuẩn hóa giá trị Hounsfield Unit (HU) về khoảng [0, 1].
    Mặc định dùng cửa sổ phổi (Lung Window): -1000 đến 400.
    """
    vol = np.clip(vol, min_hu, max_hu)
    vol = (vol - min_hu) / (max_hu - min_hu)
    return vol

def resample_volume(vol, mask, current_spacing, target_spacing=(1.0, 1.0, 1.0)):
    """
    Nội suy ảnh và mask về độ phân giải chuẩn (thường là 1mm^3).
    """
    # Tính tỷ lệ zoom
    resize_factor = np.array(current_spacing) / np.array(target_spacing)
    
    # Resample ảnh CT (Dùng nội suy tuyến tính - order 1)
    new_vol = zoom(vol, resize_factor, order=1)
    
    # Resample Mask (Dùng Nearest Neighbor - order 0 để giữ giá trị nhị phân)
    # mask.astype(float) để tránh lỗi, sau đó > 0.5 để đưa về lại boolean
    new_mask = zoom(mask.astype(float), resize_factor, order=0) > 0.5
    
    return new_vol, new_mask

def crop_roi(vol, mask, size=(64, 64, 32)):
    """
    Cắt vùng quan tâm (ROI) cố định quanh trọng tâm khối u.
    size: (Depth, Height, Width) - Ví dụ: (32, 64, 64)
    """
    # 1. Tìm trọng tâm (Center of Mass) của khối u
    z_idxs, y_idxs, x_idxs = np.where(mask)
    if len(z_idxs) == 0:
        return None, None # Mask rỗng
        
    center_z = int(np.mean(z_idxs))
    center_y = int(np.mean(y_idxs))
    center_x = int(np.mean(x_idxs))
    
    # 2. Tính toán biên cắt
    d, h, w = size
    z_min = center_z - d // 2
    y_min = center_y - h // 2
    x_min = center_x - w // 2
    
    # Xử lý biên (nếu bị âm hoặc vượt quá kích thước ảnh gốc)
    # Chúng ta sẽ dùng kỹ thuật "Pad sau, Crop trước" đơn giản hóa:
    # Cứ tính tọa độ crop, nếu lòi ra ngoài thì lát nữa pad bù vào.
    
    # Tạo một khối chứa tạm full zeros với kích thước mong muốn
    cropped_vol = np.zeros(size, dtype=vol.dtype) + (-1000) # Pad bằng giá trị khí (air)
    cropped_mask = np.zeros(size, dtype=bool)
    
    # Tính vùng giao nhau giữa hộp crop và ảnh gốc
    # Tọa độ trên ảnh gốc
    orig_z1 = max(0, z_min); orig_z2 = min(vol.shape[0], z_min + d)
    orig_y1 = max(0, y_min); orig_y2 = min(vol.shape[1], y_min + h)
    orig_x1 = max(0, x_min); orig_x2 = min(vol.shape[2], x_min + w)
    
    # Tọa độ trên ảnh crop
    crop_z1 = orig_z1 - z_min; crop_z2 = crop_z1 + (orig_z2 - orig_z1)
    crop_y1 = orig_y1 - y_min; crop_y2 = crop_y1 + (orig_y2 - orig_y1)
    crop_x1 = orig_x1 - x_min; crop_x2 = crop_x1 + (orig_x2 - orig_x1)
    
    # Copy dữ liệu vào
    if (orig_z2 > orig_z1) and (orig_y2 > orig_y1) and (orig_x2 > orig_x1):
        cropped_vol[crop_z1:crop_z2, crop_y1:crop_y2, crop_x1:crop_x2] = \
            vol[orig_z1:orig_z2, orig_y1:orig_y2, orig_x1:orig_x2]
            
        cropped_mask[crop_z1:crop_z2, crop_y1:crop_y2, crop_x1:crop_x2] = \
            mask[orig_z1:orig_z2, orig_y1:orig_y2, orig_x1:orig_x2]
            
    return cropped_vol, cropped_mask