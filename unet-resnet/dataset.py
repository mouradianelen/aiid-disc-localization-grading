import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2

TARGET_H = 384
TARGET_W = 384


class SagittalDiscDataset(Dataset):
    def __init__(self, root, subset):
        """
        root: dataset directory
        subset: 'train', 'val', or 'test'
        """
        self.root = root

        df = pd.read_csv(os.path.join(root, "overview.csv"))

        df = df[df["subset"] == subset]

        df = df[df["new_file_name"].str.contains("t2", case=False, na=False)]

        self.files = df["new_file_name"].tolist()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        fname_mha = fname if fname.endswith(".mha") else fname + ".mha"

        img_path = os.path.join(self.root, "images", fname_mha)
        msk_path = os.path.join(self.root, "masks", fname_mha)

        img_itk = sitk.ReadImage(img_path)
        msk_itk = sitk.ReadImage(msk_path)

        img_vol = sitk.GetArrayFromImage(img_itk).astype(np.float32)
        mask_vol = sitk.GetArrayFromImage(msk_itk).astype(np.int64)

       
        x_mid = img_vol.shape[2] // 2

        img_2d = img_vol[:, :, x_mid]  # (Z, Y)
        mask_2d = mask_vol[:, :, x_mid]  # (Z, Y)

        img_2d = img_2d.T
        mask_2d = mask_2d.T

        mask_2d = (mask_2d > 0).astype(np.int64)

        img_2d = (img_2d - img_2d.mean()) / (img_2d.std() + 1e-6)

        img = torch.from_numpy(img_2d).unsqueeze(0)  # (1, H, W)
        mask = torch.from_numpy(mask_2d)  # (H, W)

        img = F.interpolate(
            img.unsqueeze(0), size=(384, 384), mode="bilinear", align_corners=False
        ).squeeze(0)

        mask = (
            F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(), size=(384, 384), mode="nearest"
            )
            .squeeze(0)
            .squeeze(0)
            .long()
        )

        return img, mask


class DiscGradingDataset(Dataset):
    def __init__(self, cases, grading_csv, n_rois=10):
        self.samples = []
        self.aug = v2.Compose(
            [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomRotation(10),
            ]
        )

        df = pd.read_csv(grading_csv)

        # group grading rows by patient
        grouped = df.groupby("Patient")

        for c in cases:
            fname = c["file_name"]

            patient_id = int(fname.split("_")[0])

            if patient_id not in grouped.groups:
                continue

            rois = c["rois"]  

            patient_rows = grouped.get_group(patient_id)
            patient_rows = patient_rows.sort_values("IVD label")

            K = len(patient_rows)  # number of graded lumbar discs
            start = n_rois - K  # index where lumbar region starts

            for _, r in patient_rows.iterrows():
                ivd_label = int(r["IVD label"]) - 1  # 0-based
                disc_idx = start + ivd_label

                grade = int(r["Pfirrman grade"]) - 1  # 0–4

                self.samples.append({"image": rois[disc_idx], "grade": grade})

        print(f"[GradingDataset] Total disc samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]["image"]
        y = self.samples[idx]["grade"]
        x = self.aug(x)

        x = torch.from_numpy(x).unsqueeze(0).float()  # (1,128,128)
        y = torch.tensor(y).long()

        return x, y
