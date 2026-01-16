import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalize_mri(image_arr):
    """ MRI (uint16) converision to 0-255 (uint8)"""
    img_min = np.min(image_arr)
    img_max = np.max(image_arr)
    if img_max == img_min: return image_arr
    img_norm = (image_arr - img_min) / (img_max - img_min)
    return (img_norm * 255).astype(np.uint8)

def get_bbox_from_mask(mask_slice, class_id):
    """Finds frame (bounding box) for mask color"""
    y_indices, x_indices = np.where(mask_slice == class_id)

    if len(y_indices) == 0: return None 

    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

    # YOLO format: x_center, y_center, width, height (normalized 0-1)
    h, w = mask_slice.shape
    x_center = ((x_min + x_max) / 2) / w
    y_center = ((y_min + y_max) / 2) / h
    width = (x_max - x_min) / w
    height = (y_max - y_min) / h

    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def get_grade(patient_id, disc_idx, df_grads):
    try:
        pid = int(patient_id)
        p_rows = df_grads[df_grads['Patient'] == pid]
        target_label = disc_idx + 1
        row = p_rows[p_rows['IVD label'] == target_label]

        if not row.empty:
            val = row['Pfirrman grade'].values[0]
            if pd.notna(val): return int(val)
    except:
        return None
    return None

def imshow(inp, title=None):
    """Shows image Tensor."""
    inp = inp.numpy().transpose((1, 2, 0)) 
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # pixel = (pixel * std) + mean
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)
    if title:
        plt.title(title)
    plt.pause(0.001)