import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load once
model = tf.keras.models.load_model(
    r"C:\Prasanth\Prasanth\AllLanguages\PROJECT\brain_tumor_streamlit\models\tumor_classifier_gan_cnn.h5"
)

class_names = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

def classify_gan_cnn_image(pil_img):
    # Preprocess
    img = pil_img.resize((128, 128)).convert("L")  # grayscale
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    conf = np.max(preds)
    label = class_names[np.argmax(preds)]

    return {
        "label": label,
        "confidence": float(conf),
        "probs": {class_names[i]: float(p) for i, p in enumerate(preds[0])}
    }
