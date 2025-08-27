import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================
# Define AlexNet-like Model
# =====================
class AlexNetLike(nn.Module):
    def __init__(self, num_classes=4):
        super(AlexNetLike, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # conv1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),           # conv2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),          # conv3
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),          # conv4
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),          # conv5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),  # adjust if input size different
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# =====================
# Load Model Function
# =====================
def load_cnn_model(checkpoint_path, num_classes=4):
    model = AlexNetLike(num_classes=num_classes)
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # Load safely (ignore missing/unexpected keys & mismatches)
    new_state_dict = {}
    model_dict = model.state_dict()

    for k, v in state_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            new_state_dict[k] = v
        else:
            print(f"Ignoring layer: {k} (shape mismatch or unexpected)")

    # Update model state_dict
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict, strict=False)

    print("✅ Model loaded (mismatched layers ignored).")
    return model


# =====================
# Example Classification Function
# =====================
import torchvision.transforms as transforms
from PIL import Image

# Define preprocessing (same as during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),    # AlexNet expects 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                         std=[0.229, 0.224, 0.225]),
])

# Define your class names (adjust based on dataset)
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]  

def classify_cnn_image(img, checkpoint_path="./models/traditional_cnn.pth", num_classes=4):
    model = load_cnn_model(checkpoint_path, num_classes)
    model.eval()

    # Convert input to tensor
    if isinstance(img, Image.Image):   # if PIL image
        img_tensor = transform(img)
    elif isinstance(img, torch.Tensor):  # if already tensor
        img_tensor = img
    else:
        raise TypeError("Input img must be PIL.Image or torch.Tensor")

    with torch.no_grad():
        outputs = model(img_tensor.unsqueeze(0))  # add batch dimension
        probs = torch.softmax(outputs, dim=1)     # get probabilities
        confidence, predicted = torch.max(probs, 1)

    return {
        "class_index": predicted.item(),
        "label": CLASS_NAMES[predicted.item()],
        "confidence": confidence.item()
    }
