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

from src.models.full_model import NeuralODE3DReconstruction
from src.data.dataset_loader import LIDCDataset
from src.data.dicom_loader import DicomLoader
from src.data.preprocessing import resample_volume, crop_roi
from src.data.generation import mesh_from_mask
from src.utils.metrics import compute_all_metrics
from src.utils.marching_cubes import generate_mesh_from_sdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth file")
    args = parser.parse_args()

    # --- FIX LỖI UNICODE TẠI ĐÂY ---
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"❌ Không tìm thấy config: {args.config}")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # -------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 ĐÁNH GIÁ MÔ HÌNH TRÊN: {device}")

    # Load Model
    model = NeuralODE3DReconstruction(cfg).to(device)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"❌ Không tìm thấy checkpoint: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f"✅ Đã load checkpoint Epoch {ckpt['epoch']}")

    # Dataset & Loader
    processed_dir = cfg['paths']['processed_data']
    raw_dir = cfg['paths']['raw_data']
    test_dataset = LIDCDataset(processed_dir, split='test')

    # DicomLoader để lấy Ground Truth
    dicom_loader = DicomLoader(os.path.abspath(raw_dir))

    print(f"📦 Tập Test: {len(test_dataset)} mẫu")

    output_dir = os.path.join(cfg['paths']['output_dir'], "predictions")
    os.makedirs(output_dir, exist_ok=True)

    # Lưới tọa độ
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
        roi_tensor, _, _ = test_dataset[i]
        roi_tensor = roi_tensor.unsqueeze(0).to(device)

        # A. PREDICT
        with torch.no_grad():
            pred_sdf = model(roi_tensor, query_coords)
            pred_sdf = pred_sdf * 20.0  # Scale về mm
            pred_vol = pred_sdf.view(roi_size).cpu().numpy()

        # Tạo Mesh Dự đoán (Thử level=0.02 nếu cần)
        mesh_pred = generate_mesh_from_sdf(pred_vol, spacing=tuple(cfg['data']['target_spacing']), level=0.02)

        # B. GET GROUND TRUTH
        try:
            pid, nodule_idx_str = file_id.split('_nodule')
            nodule_idx = int(nodule_idx_str)
            vol_orig, spacing_orig, nodules = dicom_loader.load_patient_data(pid)
            annots = nodules[nodule_idx]

            mask_orig, cbbox, _ = consensus(annots, clevel=0.5, pad=10)
            vol_nodule = vol_orig[cbbox]
            _, mask_res = resample_volume(vol_nodule, mask_orig, spacing_orig, tuple(cfg['data']['target_spacing']))

            dummy_vol = np.zeros_like(mask_res, dtype=np.float32)
            _, roi_mask = crop_roi(dummy_vol, mask_res, size=roi_size)

            mesh_gt = mesh_from_mask(roi_mask, spacing=tuple(cfg['data']['target_spacing']))

        except Exception as e:
            # print(f"⚠️ Lỗi tạo GT cho {file_id}: {e}")
            mesh_gt = None

        # C. SAVE & EVALUATE
        record = {"ID": file_id}

        if mesh_pred:
            mesh_pred.apply_translation(-mesh_pred.centroid)
            save_path = os.path.join(output_dir, f"{file_id}_PRED.obj")
            mesh_pred.export(save_path)
            record["Status"] = "Success"
            record["Vertices"] = len(mesh_pred.vertices)

            if mesh_gt:
                mesh_gt.apply_translation(-mesh_gt.centroid)
                mesh_gt.export(os.path.join(output_dir, f"{file_id}_GT.obj"))
                metrics = compute_all_metrics(mesh_pred, mesh_gt)
                record.update(metrics)
        else:
            record["Status"] = "Failed"

        results.append(record)

    # Save Report
    df = pd.DataFrame(results)

    if "Chamfer" in df.columns:
        print("\n📊 KẾT QUẢ TRUNG BÌNH:")
        print(f"   🔹 Chamfer: {df['Chamfer'].mean():.4f} mm")
        print(f"   🔹 Hausdorff95: {df['Hausdorff95'].mean():.4f} mm")
        print(f"   🔹 ASSD: {df['ASSD'].mean():.4f} mm")

    csv_path = os.path.join(cfg['paths']['output_dir'], "evaluation_report.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Đã lưu báo cáo tại: {csv_path}")


if __name__ == "__main__":
    main()