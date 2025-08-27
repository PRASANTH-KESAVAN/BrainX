# densenet_cbam_loader.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# -----------------------------
# CBAM Block
# -----------------------------
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        x = x * self.sigmoid_channel(avg_out + max_out)

        # Spatial attention
        avg_sp = torch.mean(x, dim=1, keepdim=True)
        max_sp, _ = torch.max(x, dim=1, keepdim=True)
        sp = torch.cat([avg_sp, max_sp], dim=1)
        x = x * self.sigmoid_spatial(self.conv_spatial(sp))
        return x

# -----------------------------
# DenseNet169 + CBAM
# -----------------------------
class DenseNet169CBAM(nn.Module):
    def __init__(self, num_classes=4):
        super(DenseNet169CBAM, self).__init__()
        self.base_model = models.densenet169(pretrained=False)
        self.base_model_features = self.base_model.features
        self.cbam = CBAM(1664)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(1664, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.base_model_features(x)
        x = self.cbam(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.dropout1(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


# -----------------------------
# Load Model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

model = DenseNet169CBAM(num_classes=len(CLASS_NAMES)).to(device)
model.load_state_dict(torch.load(
    r"C:\Prasanth\Prasanth\AllLanguages\PROJECT\brain_tumor_streamlit\models\densenet169_cbam_final.pth",
    map_location=device
))
model.eval()

# -----------------------------
# Image Preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------
# Inference Function
# -----------------------------
def classify_densenet_cbam_image(img: Image.Image):
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())
    return {
        "label": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probs": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    }
