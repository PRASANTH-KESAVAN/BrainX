from pathlib import Path
from typing import Dict
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from config import CLASS_NAMES, IMG_SIZE, MEAN, STD, EXPLANATIONS

# Import your model
from hybrid_model import HybridResNetViT  

# Paths
BASE_DIR = Path(__file__).resolve().parent
CKPT_PATH = BASE_DIR / "models" / "hybrid_resnet50_vit_b16_best.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load checkpoint
_ckpt = torch.load(str(CKPT_PATH), map_location="cpu")
class_names = _ckpt.get("class_names", CLASS_NAMES)

# Load model
MODEL = HybridResNetViT(num_classes=len(class_names))
state = _ckpt.get("state_dict", _ckpt)
if isinstance(state, dict):
    state = {k.replace("module.", ""): v for k, v in state.items()}
    MODEL.load_state_dict(state, strict=False)
else:
    MODEL = state
MODEL.to(DEVICE).eval()

# Transforms
EVAL_TFMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def classify_image(image: Image.Image) -> Dict:
    """Classify a PIL image into tumor type."""
    x = EVAL_TFMS(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = MODEL(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    label = class_names[idx]
    top3 = sorted(
        [{"label": class_names[i], "p": float(p)} for i, p in enumerate(probs)],
        key=lambda d: d["p"], reverse=True
    )[:3]
    exp = EXPLANATIONS.get(label.lower(), "Pattern-based features consistent with predicted class.")
    return {
        "label": label,
        "confidence": float(probs[idx]),
        "top3": top3,
        "explanation": exp,
    }
