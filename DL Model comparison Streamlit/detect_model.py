from ultralytics import YOLO
from pathlib import Path
import torch
import numpy as np
from PIL import Image

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
YOLO_MODEL_PATH = Path("models/yolov8_best.pt")

# Load YOLO model once
yolo_model = YOLO(str(YOLO_MODEL_PATH))

def detect_image(image: Image.Image):
    """
    Run YOLO detection on a PIL image.
    Returns list of detections and plotted image.
    """
    np_img = np.array(image)

    results = yolo_model.predict(
        source=np_img,
        device=DEVICE,
        conf=0.25,
        imgsz=640,
        verbose=False
    )

    detections = []
    plotted_img = None
    if results and len(results) > 0:
        res = results[0]
        plotted_img = res.plot()  # numpy array with boxes
        for box in res.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()
            w = xyxy[2] - xyxy[0]
            h = xyxy[3] - xyxy[1]
            detections.append({
                "label": res.names[cls_id],
                "confidence": conf,
                "bbox": xyxy,
                "width": w,
                "height": h
            })
    return detections, plotted_img
