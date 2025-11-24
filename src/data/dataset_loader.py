import torch
from torch.utils.data import Dataset
import numpy as np
import os
import json


class LIDCDataset(Dataset):
    def __init__(self, processed_dir, split='train', split_file='split_data.json'):
        self.processed_dir = processed_dir
        self.roi_dir = os.path.join(processed_dir, "rois")
        self.sdf_dir = os.path.join(processed_dir, "sdfs")

        split_path = os.path.join(processed_dir, split_file)
        with open(split_path, 'r') as f:
            splits = json.load(f)
            self.file_ids = splits[split]

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]

        # Load ROI
        roi_path = os.path.join(self.roi_dir, f"{file_id}.npy")
        roi = np.load(roi_path).astype(np.float32)
        roi = torch.from_numpy(roi).unsqueeze(0)

        # Load SDF
        sdf_path = os.path.join(self.sdf_dir, f"{file_id}.npz")
        data = np.load(sdf_path)

        points = torch.from_numpy(data['points'].astype(np.float32))
        sdfs = torch.from_numpy(data['sdfs'].astype(np.float32)).unsqueeze(-1)

        # --- FIX QUAN TRỌNG: SCALE SDF ---
        # Chia cho 20.0 để đưa range về khoảng [-1, 1]
        sdfs = sdfs / 20.0
        # ---------------------------------

        spatial_shape = torch.tensor(roi.shape[1:]).float()
        points_norm = points / spatial_shape

        return roi, points_norm, sdfs