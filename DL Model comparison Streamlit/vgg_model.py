# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# from pathlib import Path
# from PIL import Image

# # Path to saved model
# VGG_MODEL_PATH = Path("models/vgg16_model.h5")

# # Load model once
# vgg_model = load_model(VGG_MODEL_PATH)

# # Define class names (make sure they match training order!)
# class_names = ["glioma", "meningioma", "pituitary", "no-tumor"]

# def classify_vgg(img: Image.Image):
#     """
#     Run VGG16 model on a PIL image.
#     Returns prediction dict with label, confidence, and all probabilities.
#     """
#     img = img.resize((224, 224))  # resize to model input
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0) / 255.0  # normalize [0,1]

#     preds = vgg_model.predict(x)
#     probs = preds[0]
#     idx = int(np.argmax(probs))
#     label = class_names[idx]

#     top3_idx = probs.argsort()[-3:][::-1]
#     top3 = [{"label": class_names[i], "p": float(probs[i])} for i in top3_idx]

#     return {
#         "label": label,
#         "confidence": float(probs[idx]),
#         "top3": top3,
#     }
