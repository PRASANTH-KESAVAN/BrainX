# app/mcp/tools/classify.py
from typing import Dict
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

# Import your model class; support both run modes:
#  - uvicorn mcp.orchestrator:app   (inside app/)
#  - uvicorn app.mcp.orchestrator:app (from parent)
try:
    from app.hybrid_model import HybridResNetViT
except ImportError:
    from hybrid_model import HybridResNetViT

BASE_DIR = Path(__file__).resolve().parents[2]   # → app/
MODELS_DIR = BASE_DIR / "models"
CKPT_PATH = MODELS_DIR / "hybrid_resnet50_vit_b16_best.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load checkpoint
_ckpt = torch.load(str(CKPT_PATH), map_location="cpu")
class_names = _ckpt.get("class_names") or ["glioma","meningioma","pituitary","no-tumor"]
img_size    = int(_ckpt.get("img_size", 224))
mean        = _ckpt.get("mean", [0.485,0.456,0.406])
std         = _ckpt.get("std",  [0.229,0.224,0.225])

MODEL = HybridResNetViT(num_classes=len(class_names))
state = _ckpt.get("state_dict", _ckpt)
if isinstance(state, dict):
    state = {k.replace("module.",""): v for k,v in state.items()}
    MODEL.load_state_dict(state, strict=False)
else:
    MODEL = state
MODEL.to(DEVICE).eval()

EVAL_TFMS = transforms.Compose([
    transforms.Resize((img_size,img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean,std),
])

EXPLANATIONS = {
    "pituitary":  "Sellar/suprasellar lesion; smooth margins; possible mass effect on the optic chiasm.",
    "meningioma": "Extra-axial dural-based mass with broad dural attachment and smooth margins.",
    "glioma":     "Intra-axial infiltrative lesion; T2/FLAIR hyperintensity with ill-defined margins.",
    "no-tumor":   "No convincing mass-like enhancement or extra-axial lesion pattern seen.",
    "notumor":    "No convincing mass-like enhancement or extra-axial lesion pattern seen."
}

def classify_image(image_path: str) -> Dict:
    img = Image.open(image_path).convert("RGB")
    x = EVAL_TFMS(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = MODEL(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    label = class_names[idx]
    top3 = sorted(
        [{"label": class_names[i], "p": float(p)} for i,p in enumerate(probs)],
        key=lambda d: d["p"], reverse=True
    )[:3]
    lab_norm = label.lower().replace(" ","").replace("-","").replace("_","")
    exp = EXPLANATIONS.get(lab_norm, "Pattern-based features consistent with predicted class.")
    return {
        "label": label,
        "confidence": float(probs[idx]),
        "top3": top3,
        "explanation": exp,
    }
