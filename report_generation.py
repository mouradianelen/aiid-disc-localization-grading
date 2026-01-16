import SimpleITK as sitk
import numpy as np
import os
import torch.nn.functional as F
import scipy.ndimage as ndi
import matplotlib.patches as patches
import torch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
import matplotlib.pyplot as plt


def get_candidates_from_mask(mask):
    labeled, _ = ndi.label(mask == 1)
    slices_all = ndi.find_objects(labeled)

    candidates = []

    for lbl, slc in enumerate(slices_all, start=1):
        if slc is None:
            continue

        h = slc[0].stop - slc[0].start
        w = slc[1].stop - slc[1].start
        area = (labeled == lbl).sum()

        # SAME filtering you trusted before
        if area < 50:
            continue
        if h < 8 or w < 8:
            continue

        cy = (slc[0].start + slc[0].stop) / 2

        candidates.append({"slice": slc, "cy": cy, "area": area})

    return candidates


def select_discs_by_position(candidates, n_discs=10):
    candidates = sorted(candidates, key=lambda c: c["cy"])

    if len(candidates) == 0:
        return []

    if len(candidates) >= n_discs:
        idxs = np.linspace(0, len(candidates) - 1, n_discs).round().astype(int)
        selected = [candidates[i] for i in idxs]
    else:
        selected = candidates.copy()
        while len(selected) < n_discs:
            selected.append(selected[-1])

    return selected


def slices_to_boxes(selected):
    boxes = []
    for c in selected:
        slc = c["slice"]
        y0, y1 = slc[0].start, slc[0].stop
        x0, x1 = slc[1].start, slc[1].stop
        boxes.append((y0, y1, x0, x1))
    return boxes


def load_sagittal_slice(root, file_name):
    path = os.path.join(root, "images", file_name + ".mha")
    itk = sitk.ReadImage(path)
    vol = sitk.GetArrayFromImage(itk)  

    x = vol.shape[2] // 2
    img = vol[:, :, x]

    img = img.astype(np.float32)
    img = (img - img.mean()) / (img.std() + 1e-6)
    return img


def segment_discs(img, seg_model, device):
    """
    img: (H, W) numpy array in original image space

    Returns:
        pred_384 : (384, 384) binary mask in UNet space
        scale_y  : scaling factor to map back to img
        scale_x  : scaling factor to map back to img
    """
    H, W = img.shape

    x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(device)
    x = F.interpolate(x, size=(384, 384), mode="bilinear", align_corners=False)

    with torch.no_grad():
        out = seg_model(x)
        pred_384 = torch.argmax(out, dim=1).squeeze().cpu().numpy()

    scale_y = H / 384.0
    scale_x = W / 384.0

    return pred_384.astype(np.uint8), scale_y, scale_x


def extract_disc_boxes(mask, n_discs=10, min_area=200):
    labeled, _ = ndi.label(mask)
    objects = ndi.find_objects(labeled)

    candidates = []
    for i, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        area = (labeled == i).sum()
        if area < min_area:
            continue
        cy = (slc[0].start + slc[0].stop) / 2
        candidates.append((cy, slc))

    candidates.sort(key=lambda x: x[0])

    if len(candidates) >= n_discs:
        idxs = np.linspace(0, len(candidates) - 1, n_discs).round().astype(int)
        selected = [candidates[i][1] for i in idxs]
    else:
        selected = [c[1] for c in candidates]
        while len(selected) < n_discs:
            selected.append(selected[-1])

    boxes = []
    for slc in selected:
        y0, y1 = slc[0].start, slc[0].stop
        x0, x1 = slc[1].start, slc[1].stop
        boxes.append((y0, y1, x0, x1))

    return boxes


def visualize_case_with_grades(img, boxes, gt_grades, model, device, title=""):
    """
    img       : (H, W) numpy array (original sagittal slice)
    boxes     : list of bounding boxes [(y0, y1, x0, x1), ...] length = N
    gt_grades : list of ground truth grades (0-4) or None
    """

    model.eval()

    fig, ax = plt.subplots(1, figsize=(6, 8))
    ax.imshow(img, cmap="gray")

    for i, (y0, y1, x0, x1) in enumerate(boxes):
        roi = img[y0:y1, x0:x1]

        roi_t = torch.from_numpy(roi).unsqueeze(0).unsqueeze(0).float()
        roi_t = torch.nn.functional.interpolate(
            roi_t, size=(128, 128), mode="bilinear", align_corners=False
        )
        roi_t = roi_t.to(device)

        with torch.no_grad():
            pred = model(roi_t).argmax(dim=1).item()

        rect = patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)

        if gt_grades is not None:
            label = f"D{i} | {gt_grades[i]+1} → {pred+1}"
        else:
            label = f"D{i} | Pred: {pred+1}"

        ax.text(
            x0,
            y0 - 5,
            label,
            color="red",
            fontsize=9,
            bbox=dict(facecolor="black", alpha=0.6, pad=1),
        )

    ax.set_title(title)
    ax.axis("off")
    plt.show()


def grade_rois(img, boxes, device, grade_model):
    preds = []

    for y0, y1, x0, x1 in boxes:
        roi = img[y0:y1, x0:x1]
        roi = torch.from_numpy(roi).unsqueeze(0).unsqueeze(0).float()
        roi = F.interpolate(roi, size=(128, 128), mode="bilinear", align_corners=False)
        roi = roi.to(device)

        with torch.no_grad():
            pred = grade_model(roi).argmax(dim=1).item()

        preds.append(pred)

    return preds


def visualize_full_pipeline(img, boxes, preds, title=""):
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.imshow(img, cmap="gray")

    for i, (y0, y1, x0, x1) in enumerate(boxes):
        rect = patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)

        ax.text(
            x0,
            y0 - 5,
            f"D{i} → {preds[i] + 1}",
            color="red",
            fontsize=9,
            bbox=dict(facecolor="black", alpha=0.6),
        )

    ax.set_title(title)
    ax.axis("off")
    plt.show()


def compute_disc_metrics(boxes):
    heights = [(y1 - y0) for (y0, y1, x0, x1) in boxes]
    mean_height = np.mean(heights)

    metrics = []
    for i, (y0, y1, x0, x1) in enumerate(boxes):
        height = y1 - y0
        depth = x1 - x0

        metrics.append(
            {
                "disc": i,  
                "height_px": height,
                "depth_px": depth,
                "hd_ratio": height / (depth + 1e-6),
                "height_index": height / (mean_height + 1e-6),
            }
        )

    return metrics


def generate_case_pdf(case_id, overlay_img_path, metrics, out_pdf_path):
    doc = SimpleDocTemplate(
        out_pdf_path,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    elements = []

    elements.append(
        Paragraph(f"<b>Intervertebral Disc Analysis Report</b>", styles["Title"])
    )
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(f"<b>Case ID:</b> {case_id}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    elements.append(
        Paragraph(
            "<b>Sagittal MRI with Disc Localization and Grading</b>", styles["Heading2"]
        )
    )
    elements.append(Spacer(1, 8))

    img = Image(overlay_img_path, width=300, height=420)
    elements.append(img)
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("<b>Quantitative Disc Metrics</b>", styles["Heading2"]))
    elements.append(Spacer(1, 8))

    table_data = [["Disc", "Height (px)", "Depth (px)", "H/D Ratio", "Height Index"]]

    for m in metrics:
        table_data.append(
            [
                f"D{m['disc']}",
                f"{m['height_px']:.1f}",
                f"{m['depth_px']:.1f}",
                f"{m['hd_ratio']:.3f}",
                f"{m['height_index']:.3f}",
            ]
        )

    table = Table(table_data, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ("TOPPADDING", (0, 0), (-1, 0), 8),
            ]
        )
    )

    elements.append(table)

    doc.build(elements)


def save_overlay_image(img, boxes, preds_grade, out_path, title=""):
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.imshow(img, cmap="gray")

    for d, (y0, y1, x0, x1) in enumerate(boxes):
        rect = patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            x0,
            y0 - 5,
            f"D{d} | G{preds_grade[d] + 1}",
            color="red",
            fontsize=9,
            bbox=dict(facecolor="black", alpha=0.6, pad=1),
        )

    ax.set_title(title)
    ax.axis("off")

    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
