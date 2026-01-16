import torch
import cv2
import numpy as np
from ultralytics import YOLO
from torchvision import models, transforms
from PIL import Image

class SpineAnalyzerPipeline:
    def __init__(self, models_dir='./spine_ai_system', device=None):
        """
        Loading models and running pipelines
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading pipeline on {self.device}...")

        # 1. YOLO
        self.yolo = YOLO(f"{models_dir}/spine_localizer_yolo.pt")

        # 2. EfficientNet
        checkpoint = torch.load(f"{models_dir}/spine_grader_effnet.pth", map_location=self.device)
        self.class_names = checkpoint['class_names']
        self.grader = models.efficientnet_b0(weights=None)
        # head
        num_classes = len(self.class_names)
        self.grader.classifier[1] = torch.nn.Linear(self.grader.classifier[1].in_features, num_classes)
        
        # Wages load
        self.grader.load_state_dict(checkpoint['state_dict'])
        self.grader.to(self.device)
        self.grader.eval()
        
        # Transformations as in validation
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.disc_id = 2
        for k, v in self.yolo.names.items():
            if 'disc' in v.lower(): self.disc_id = k

    def analyze_image(self, image_path):
        """
        Perform results
        """
        img = cv2.imread(image_path)
        if img is None: raise ValueError("Image not found")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        # A. Detection
        results = self.yolo.predict(img, conf=0.25, verbose=False)[0]
        boxes = results.boxes.data.cpu().numpy()
        
        # Disc filtering
        discs = [b for b in boxes if int(b[5]) == self.disc_id]
        # Sorting
        discs.sort(key=lambda x: x[1])
        
        output_data = []

        # B. Classification
        for i, box in enumerate(discs):
            x1, y1, x2, y2 = map(int, box[:4])
            
            pad_x, pad_y = int((x2-x1)*0.1), int((y2-y1)*0.1)
            cx1, cx2 = max(0, x1-pad_x), min(w, x2+pad_x)
            cy1, cy2 = max(0, y1-pad_y), min(h, y2+pad_y)
            
            crop = img_rgb[cy1:cy2, cx1:cx2]
            if crop.size == 0: continue
            
            # EfficientNet Inference
            pil_img = Image.fromarray(crop)
            img_t = self.transforms(pil_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                out = self.grader(img_t)
                prob = torch.nn.functional.softmax(out, dim=1)
                score, idx = torch.max(prob, 1)
            
            predicted_grade = self.class_names[idx.item()]
            confidence = score.item()
            
            output_data.append({
                'disc_label': f"D{i+1}",
                'grade': predicted_grade,
                'confidence': f"{confidence:.2f}",
                'bbox': [x1, y1, x2, y2]
            })
            
        return output_data

# === RUN ===
#!unzip spine_ai_pipeline.zip -d spine_ai_system

pipeline = SpineAnalyzerPipeline('./spine_ai_system')
wyniki = pipeline.analyze_image('pacjent_image.jpg')
print(wyniki)