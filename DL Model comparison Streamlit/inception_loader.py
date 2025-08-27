import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# Load model
model = load_model("models/inception_model.h5")

# Define class names (must match your training dataset order)
CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

# Preprocessing function
def preprocess_image(img: Image.Image):
    img = img.convert("RGB").resize((224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = img_array / 255.0   # rescale (since you used ImageDataGenerator(rescale=1./255))
    return np.expand_dims(img_array, axis=0)

# Prediction function
def classify_inception_image(img: Image.Image):
    img_t = preprocess_image(img)
    preds = model.predict(img_t, verbose=0)[0]

    pred_idx = int(np.argmax(preds))
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(preds[pred_idx])

    # Top-3 predictions
    top3_idx = np.argsort(preds)[::-1][:3]
    top3 = [{"label": CLASS_NAMES[i], "p": float(preds[i])} for i in top3_idx]

    return {
        "label": pred_label,
        "confidence": confidence,
        "top3": top3
    }
