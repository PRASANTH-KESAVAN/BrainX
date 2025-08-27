# effnet_model_loader.py
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# Define number of classes
num_classes = 4  # adjust if you trained with different number

# Load EfficientNet
def load_effnet_model(model_path="models/effnet_model.pt"):
    model = models.efficientnet_b0(pretrained=False)
    # Replace classifier for your number of classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    # Load weights
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

# Preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])   # same as training
])

# Classification function
def classify_effnet_image(image_path, model_path="models/effnet_model.pt"):
    model = load_effnet_model(model_path)
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
    return pred_class, probs.numpy().tolist()[0]
