import os
import yaml
import torch
import numpy as np
import pandas as pd
import trimesh
from tqdm import tqdm
import argparse
import pylidc as pl
from pylidc.utils import consensus

# Import các module dự án
from src.models.full_model import NeuralODE3DReconstruction
from src.data.dataset_loader import LIDCDataset
from src.data.dicom_loader import DicomLoader
from src.data.preprocessing import resample_volume, crop_roi
from src.data.generation import mesh_from_mask
from src.utils.metrics import compute_all_metrics
from src.utils.marching_cubes import generate_mesh_from_sdf


def main():
    # --- 1. SETUP ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 ĐÁNH GIÁ MÔ HÌNH TRÊN: {device}")

    # --- 2. LOAD MODEL ---
    model = NeuralODE3DReconstruction(cfg).to(device)
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"❌ Không tìm thấy: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f"✅ Đã load checkpoint Epoch {ckpt['epoch']}")

    # --- 3. DATASET & LOADER ---
    processed_dir = cfg['paths']['processed_data']
    raw_dir = cfg['paths']['raw_data']
    test_dataset = LIDCDataset(processed_dir, split='test')

    # Cần DicomLoader để lấy Mask gốc làm Ground Truth
    dicom_loader = DicomLoader(os.path.abspath(raw_dir))

    print(f"📦 Tập Test: {len(test_dataset)} mẫu")

    # Tạo thư mục lưu
    output_dir = os.path.join(cfg['paths']['output_dir'], "predictions")
    os.makedirs(output_dir, exist_ok=True)

    # Chuẩn bị lưới tọa độ (Cache sẵn để dùng lại)
    roi_size = cfg['data']['roi_size']
    z = torch.linspace(0, 1, roi_size[0])
    y = torch.linspace(0, 1, roi_size[1])
    x = torch.linspace(0, 1, roi_size[2])
    grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')
    query_coords = torch.stack([grid_z, grid_y, grid_x], dim=-1).reshape(-1, 3).unsqueeze(0).to(device)

    results = []
    print("🔥 Bắt đầu suy luận & đánh giá...")

    for i in tqdm(range(len(test_dataset))):
        file_id = test_dataset.file_ids[i]

        # --- A. PREDICT (Dự đoán) ---
        roi_tensor, _, _ = test_dataset[i]
        roi_tensor = roi_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            pred_sdf = model(roi_tensor, query_coords)
            pred_sdf = pred_sdf * 20.0  # Scale về mm
            pred_vol = pred_sdf.view(roi_size).cpu().numpy()

        # Tạo Mesh Dự đoán
        mesh_pred = generate_mesh_from_sdf(
            pred_vol,
            spacing=tuple(cfg['data']['target_spacing']),
            level=0.0
        )

        # --- B. GET GROUND TRUTH (Lấy đáp án thật) ---
        # Phải load lại từ DicomLoader vì prepare_data không lưu mask gốc
        try:
            pid, nodule_idx_str = file_id.split('_nodule')
            nodule_idx = int(nodule_idx_str)

            vol_orig, spacing_orig, nodules = dicom_loader.load_patient_data(pid)
            annots = nodules[nodule_idx]

            # Tái tạo quy trình Preprocessing để có Mask chuẩn
            mask_orig, cbbox, _ = consensus(annots, clevel=0.5, pad=10)
            vol_nodule = vol_orig[cbbox]  # Cần vol để resample đồng bộ
            _, mask_res = resample_volume(vol_nodule, mask_orig, spacing_orig, tuple(cfg['data']['target_spacing']))

            # Crop ROI (Chỉ lấy mask)
            dummy_vol = np.zeros_like(mask_res, dtype=np.float32)
            _, roi_mask = crop_roi(dummy_vol, mask_res, size=roi_size)

            # Tạo Mesh Ground Truth
            mesh_gt = mesh_from_mask(roi_mask, spacing=tuple(cfg['data']['target_spacing']))

        except Exception as e:
            print(f"⚠️ Lỗi tạo GT cho {file_id}: {e}")
            mesh_gt = None

        # --- C. EVALUATE & SAVE ---
        record = {"ID": file_id}

        if mesh_pred and mesh_gt:
            # Căn giữa cả 2 để so sánh hình dáng (bỏ qua sai lệch vị trí tịnh tiến)
            mesh_pred.apply_translation(-mesh_pred.centroid)
            mesh_gt.apply_translation(-mesh_gt.centroid)

            # Tính Metric
            metrics = compute_all_metrics(mesh_pred, mesh_gt)
            record.update(metrics)  # Chamfer, Hausdorff, ASSD
            record["Status"] = "Success"

            # Lưu file 3D
            mesh_pred.export(os.path.join(output_dir, f"{file_id}_PRED.obj"))
            mesh_gt.export(os.path.join(output_dir, f"{file_id}_GT.obj"))  # Lưu luôn GT để so sánh

        else:
            record["Status"] = "Failed"
            # Ghi chú lý do
            if not mesh_pred:
                record["Note"] = "Pred Empty"
            elif not mesh_gt:
                record["Note"] = "GT Error"

        results.append(record)

    # --- 5. BÁO CÁO TỔNG HỢP ---
    df = pd.DataFrame(results)

    # Tính trung bình các chỉ số
    if "Chamfer" in df.columns:
        print("\n📊 KẾT QUẢ TRUNG BÌNH TRÊN TẬP TEST:")
        print(f"   🔹 Chamfer Distance: {df['Chamfer'].mean():.4f} mm")
        print(f"   🔹 Hausdorff (95%):  {df['Hausdorff95'].mean():.4f} mm")
        print(f"   🔹 ASSD:             {df['ASSD'].mean():.4f} mm")

    csv_path = os.path.join(cfg['paths']['output_dir'], "evaluation_report.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Đã lưu báo cáo chi tiết tại: {csv_path}")


if __name__ == "__main__":
    main()