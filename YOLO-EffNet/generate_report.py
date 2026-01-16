import os
import cv2
import torch
import numpy as np
from PIL import Image as PILImage
from fpdf import FPDF
from torchvision import transforms
from datetime import datetime

# === Configuration for EFFICIENTNET ===
grading_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# pfirmann classes
GRADE_CLASSES = ['G1', 'G2', 'G3', 'G4', 'G5']

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Intervertebral Disc Analysis Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")} | AI Assisted Diagnosis', 0, 0, 'C')

def generate_medical_report(image_path, yolo_model, grading_model, output_pdf_path="report.pdf"):
    """
    Creates report
    """
    # 1. Image
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        print(f"Błąd: Nie można wczytać {image_path}")
        return
    
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    h_img, w_img, _ = img_cv.shape
    case_id = os.path.basename(image_path).split('.')[0]

    # 2. YOLO detection
    print(f"-> Analiza YOLO: {case_id}...")
    results = yolo_model.predict(image_path, conf=0.25, verbose=False)[0]
    boxes = results.boxes.data.cpu().numpy()
    disc_id = 2
    for k, v in yolo_model.names.items():
        if 'disc' in v.lower(): disc_id = k

    # sorting
    disc_boxes = [b for b in boxes if int(b[5]) == disc_id]
    disc_boxes.sort(key=lambda x: x[1])

    if not disc_boxes:
        print("Nie znaleziono dysków!")
        return
    disc_metrics = []
    
    # 3. Grading + Drawing
    print(f"-> Klasyfikacja EfficientNet dla {len(disc_boxes)} dysków...")
    grading_model.eval()
    annotated_img = img_cv.copy()

    with torch.no_grad():
        for i, box in enumerate(disc_boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            pad_x = int((x2-x1)*0.1)
            pad_y = int((y2-y1)*0.1)
            cx1 = max(0, x1-pad_x); cx2 = min(w_img, x2+pad_x)
            cy1 = max(0, y1-pad_y); cy2 = min(h_img, y2+pad_y)

            # (Crop)
            crop = img_rgb[cy1:cy2, cx1:cx2]
            pil_crop = PILImage.fromarray(crop)
            
            # Transform and pred
            input_tensor = grading_transforms(pil_crop).unsqueeze(0).to(DEVICE)
            outputs = grading_model(input_tensor)
            _, pred_idx = torch.max(outputs, 1)
            grade = GRADE_CLASSES[pred_idx.item()]
            height_px = y2 - y1
            depth_px = x2 - x1
            ratio = height_px / depth_px if depth_px > 0 else 0
            
            # Save
            disc_label = f"D{i+1}"
            disc_metrics.append([disc_label, grade, f"{height_px} px", f"{depth_px} px", f"{ratio:.3f}"])

            # Drawing on image
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label_text = f"{disc_label} | {grade}"
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated_img, (x1, y1 - 20), (x1 + w, y1), (0, 0, 255), -1)
            cv2.putText(annotated_img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    temp_img_path = "temp_annotated.jpg"
    cv2.imwrite(temp_img_path, annotated_img)

    # 4. PDF
    print("->  PDF...")
    pdf = PDFReport()
    pdf.add_page()
    
    # Info section
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, f'Case ID: {case_id}', 0, 1)
    pdf.ln(5)
    
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Sagittal MRI with Disc Localization and Grading', 0, 1)
    pdf.ln(5)

    # Put image
    pdf.image(temp_img_path, x=35, w=140) 
    pdf.ln(10)

    # Table
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Quantitative Disc Metrics', 0, 1)
    pdf.ln(5)

    pdf.set_font('Arial', 'B', 10)
    pdf.set_fill_color(200, 200, 200) 
    col_width = 35
    headers = ['Disc ID', 'Pfirrmann Grade', 'Height', 'Depth', 'H/D Ratio']
    
    for header in headers:
        pdf.cell(col_width, 10, header, 1, 0, 'C', 1)
    pdf.ln()
    pdf.set_font('Arial', '', 10)
    for row in disc_metrics:
        for item in row:
            pdf.cell(col_width, 10, str(item), 1, 0, 'C')
        pdf.ln()

    pdf.output(output_pdf_path)
    print(f"✅ Report ready: {output_pdf_path}")
    
    from IPython.display import display, Image
    display(Image(filename=temp_img_path))

# === RUN ===
import glob
val_images = glob.glob('./datasets/spider_yolo/val/images/*.jpg')
test_img_path = val_images[0]
effnet = effnet.to(DEVICE)

generate_medical_report(test_img_path, model, effnet, "Pacient_report.pdf")