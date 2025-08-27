#vgg_model_loader.py
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Path to your saved VGG16 model
VGG16_MODEL_PATH = "./models/vgg16_model1.keras"

# Load the model once (cached)
vgg16_model = load_model(VGG16_MODEL_PATH)

# Define class labels (update based on your dataset)
CLASS_LABELS = ["glioma", "meningioma", "notumor", "pituitary"]

# ---------------------------
# Function to preprocess & classify
# ---------------------------
def classify_vgg16_image(img):
    """
    Classify an image using trained VGG16 model
    Args:
        img: PIL image uploaded via Streamlit
    Returns:
        predicted_label, confidence
    """
    img = img.resize((224, 224))   # VGG16 input size
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = vgg16_model.predict(img_array)
    pred_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds)

    return CLASS_LABELS[pred_class], confidence
