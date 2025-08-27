# mobilenet_loader.py (ONNX Runtime version)
import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Load ONNX session
ort_session = ort.InferenceSession("models/mobilenet_brain.onnx")

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

CLASS_NAMES = ["glioma", "meningioma", "pituitary", "noTumor"]

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img = transform(image).unsqueeze(0)  # shape: (1, 3, 224, 224)
    img = img.numpy()
    img = np.transpose(img, (0, 2, 3, 1))  # change to (1, 224, 224, 3) for ONNX
    return img.astype(np.float32)

def classify_mobilenet_image(image):
    img_t = preprocess_image(image)
    ort_inputs = {ort_session.get_inputs()[0].name: img_t}
    ort_outs = ort_session.run(None, ort_inputs)

    probs = np.array(ort_outs[0][0])  # first batch element
    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    return {"label": pred_label, "confidence": confidence}
