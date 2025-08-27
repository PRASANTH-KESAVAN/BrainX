# -*- coding: utf-8 -*-
"""
Brain MRI classifier + MedGema SLM report generator
- Loads HybridResNetViT checkpoint to classify tumor type
- Calls amsaravi/medgemma-4b-it:q6 via Ollama to generate a structured report
- Includes a safe import shim that stubs torchmetrics if it's missing/broken
"""

import os
import sys
import types
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import ollama

# -------------------------
# 🖼️ Your image path
# -------------------------
IMG_PATH = r"C:\Users\Niranjan\Downloads\NPN\Hacakathon\WhatsApp + MCP + MedGemma\Data_Standardized\Validation\meningioma\Tr-me_1270.jpg"

# -------------------------
# ⚙️ Checkpoint path (adjust if needed)
# -------------------------
CHECKPOINT_PATH = "models/hybrid_resnet50_vit_b16_best.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _safe_import_hybrid_model():
    """
    Try importing HybridResNetViT. If torchmetrics/torchaudio cause import errors,
    install a tiny stub for torchmetrics.classification so inference can proceed.
    """
    try:
        from hybrid_model import HybridResNetViT  # noqa: F401
        return HybridResNetViT
    except Exception:
        # Create a stub torchmetrics with the classification submodule
        tm = types.ModuleType("torchmetrics")
        tm_class = types.ModuleType("torchmetrics.classification")

        class _DummyMetric:
            def __init__(self, *args, **kwargs): pass
            def update(self, *args, **kwargs): pass
            def compute(self): return 0.0
            def reset(self): pass

        for name in [
            "MulticlassAccuracy",
            "MulticlassPrecision",
            "MulticlassRecall",
            "MulticlassF1Score",
            "MulticlassAUROC",
        ]:
            setattr(tm_class, name, _DummyMetric)

        tm.classification = tm_class
        sys.modules["torchmetrics"] = tm
        sys.modules["torchmetrics.classification"] = tm_class

        # Retry import with stubs in place
        from hybrid_model import HybridResNetViT  # type: ignore
        return HybridResNetViT


# Import model class safely
HybridResNetViT = _safe_import_hybrid_model()


def load_model_and_transforms(ckpt_path: str) -> Tuple[torch.nn.Module, transforms.Compose, list]:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    num_classes = int(checkpoint.get("num_classes", 4))
    model = HybridResNetViT(num_classes=num_classes)

    state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state, strict=False)
    model = model.to(DEVICE).eval()

    class_names = checkpoint.get("class_names", ["glioma", "meningioma", "no_tumor", "pituitary"])
    img_size = int(checkpoint.get("img_size", 224))
    mean = checkpoint.get("mean", [0.485, 0.456, 0.406])
    std = checkpoint.get("std", [0.229, 0.224, 0.225])

    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return model, eval_tfms, class_names


def load_rgb_image(path: str) -> Image.Image:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return img


def predict_image(model: torch.nn.Module, tfms: transforms.Compose, img: Image.Image, class_names: list):
    x = tfms(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    label = class_names[idx] if 0 <= idx < len(class_names) else f"class_{idx}"
    confidence = float(probs[idx])
    return label, confidence


def normalize_tumor_label(raw_label: str) -> str:
    low = raw_label.lower()
    if "glioma" in low:
        return "Glioma"
    if "meningioma" in low or "meningiomas" in low:
        return "Meningioma"
    if "pituitary" in low:
        return "Pituitary Adenoma"
    if "no" in low and "tumor" in low:
        return "No Tumor"
    return raw_label.title()


def build_medgema_prompt(tumor_name: str) -> str:
    prompt = f"""
Analyze the attached brain MRI image that has been automatically classified as **{tumor_name}**.
You are NOT providing a medical diagnosis; produce only an **AI-based image research description**.

Return a **structured, numbered, Markdown report** with the following sections:

1. **Tumor Type Classification** – Identify/confirm the specific tumor type (e.g., glioma, meningioma, pituitary adenoma, etc.) based on the image features and common patterns for **{tumor_name}**.
2. **Tumor Size Estimation** – Estimate tumor **diameter in centimeters [cm]** (approximate from the image; state assumptions about scale if unknown).
3. **Tumor Grading & Risk Stratification** – Provide general literature-based guidance on low-grade vs high-grade tendencies for **{tumor_name}**, and associated clinical risk levels (not a diagnosis).
4. **Personalized Treatment Recommendations (Research Perspective)** – Outline typical options and pathways considering patient-specific factors (age, sex, medical history, comorbidities) in a general, non-patient-specific way.
5. **Symptom Analysis & Supportive Remedies** – List likely symptoms associated with **{tumor_name}** and supportive remedies to alleviate discomfort.
6. **Functional Impact Assessment** – Discuss potential neurological/physical impairments (e.g., motor dysfunction, visual impairment, speech or cognitive deficits).
7. **Recurrence Risk Prediction** – Summarize general trends in recurrence risk after standard treatments for **{tumor_name}**.
8. **Benign–Malignant Prediction** – Provide general likelihood tendencies of **benign vs malignant** behavior for **{tumor_name}**, citing typical patterns (not patient-specific).
9. **Comorbidity Analysis** – Describe common comorbidities that can coexist with this tumor type and how they may affect diagnosis, treatment, and outcomes.

**Formatting requirements:**
- Use H2/H3 headings for each numbered section.
- Where appropriate, include short bullet points.
- Keep tone scientific and concise.
- Add a brief final disclaimer reminding this is not medical advice or a diagnosis.
"""
    return prompt.strip()


def call_medgema_report(image_path: str, tumor_name: str) -> str:
    prompt = build_medgema_prompt(tumor_name)
    response = ollama.chat(
        model="amsaravi/medgemma-4b-it:q6",
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [image_path],
        }],
    )
    return response["message"]["content"]


def main():
    print("🔧 Loading model & transforms...")
    model, eval_tfms, class_names = load_model_and_transforms(CHECKPOINT_PATH)

    print(f"🖼️ Loading image: {IMG_PATH}")
    img = load_rgb_image(IMG_PATH)

    print("🔎 Running classification...")
    raw_label, conf = predict_image(model, eval_tfms, img, class_names)
    tumor_name = normalize_tumor_label(raw_label)
    print(f"✅ Prediction: {tumor_name} (confidence: {conf:.2f})")

    if tumor_name.lower() == "no tumor":
        print("\n🎉 No tumor detected by the classifier. Have a good day :)")
        return

    print("\n🧠 Generating MedGema research-style report (items 1–9)...")
    try:
        report = call_medgema_report(IMG_PATH, tumor_name)
    except Exception as e:
        print("❌ Error calling MedGema SLM via Ollama:", e)
        sys.exit(1)

    print("\n===================== 📄 RESEARCH REPORT 📄 =====================")
    print(report)
    print("\n=================================================================")
    print("\n⚠️ Disclaimer: This is NOT medical advice or a diagnosis. "
          "It is an AI-generated, literature-style description based on the provided image.")


if __name__ == "__main__":
    main()
