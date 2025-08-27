# config.py

CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no-tumor"]

IMG_SIZE = 224  # default size for ResNet+ViT
CNN_IMG_SIZE = 128  # CNN size

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

EXPLANATIONS = {
    "pituitary":  "Sellar/suprasellar lesion; smooth margins; possible mass effect on the optic chiasm.",
    "meningioma": "Extra-axial dural-based mass with broad dural attachment and smooth margins.",
    "glioma":     "Intra-axial infiltrative lesion; T2/FLAIR hyperintensity with ill-defined margins.",
    "no-tumor":   "No convincing mass-like enhancement or extra-axial lesion pattern seen.",
}
