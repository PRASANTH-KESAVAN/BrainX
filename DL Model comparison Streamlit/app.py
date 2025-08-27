import streamlit as st
from PIL import Image
import pandas as pd

# Import model functions
from model import classify_image                  # ResNet+ViT
from detect_model import detect_image             # YOLO
from cnn_model_loader import classify_cnn_image   # CNN
from vgg_model_loader import classify_vgg16_image # VGG16
from mobilenet_loader import classify_mobilenet_image  # MobileNet
from effnet_model_loader import classify_effnet_image  # EfficientNet
from inception_loader import classify_inception_image  # Inception (new)
from densenet_cbam_loader import classify_densenet_cbam_image
from gan_cnn_loader import classify_gan_cnn_image



st.set_page_config(page_title="SmartMed AI", layout="wide")

st.title("🧠 SmartMed AI: Brain Tumor Classifier Comparison")
st.markdown("""
"Discover how different deep learning models see the same problem. 
Our tool compares ResNet, MobileNet, EfficientNet, YOLO, Inception, and more, giving you clear insights into their performance on brain tumor detection."
""")

# ------------------- Sidebar -------------------
st.sidebar.header("⚙️ DL Models")
mode = st.sidebar.radio(
    "Choose Model",
    [
        "ResNet+ViT",
        "YOLO",
        "CNN (Traditional)",
        "VGG16",
        "MobileNet",
        "EfficientNet",
        "DenseNet169+CBAM",
        "Inception (Ensemble)",
        "GAN+CNN",
        "Run All Models"
    ]
)

st.sidebar.info("Upload a Brain MRI image and test across different DL models.")

# ------------------- File Uploader -------------------
uploaded_file = st.file_uploader("📤 Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Uploaded MRI Image", width=350)

    results = {}

    # ------------------- Single Model Prediction -------------------
    if mode == "ResNet+ViT":
        with st.spinner("🔍 Running ResNet+ViT..."):
            result = classify_image(img)
        results["ResNet+ViT"] = [result['label'], f"{result['confidence']:.2f}"]

        st.markdown("### 📊 Top-3 Predictions (ResNet+ViT)")
        df_top3 = pd.DataFrame(result["top3"])
        df_top3["p"] = df_top3["p"].round(2)
        st.table(df_top3.rename(columns={"label": "Class", "p": "Confidence"}))

    elif mode == "YOLO":
        with st.spinner("🔍 Running YOLO..."):
            detections, plotted_img = detect_image(img)
        if detections:
            results["YOLO"] = [detections[0]['label'], f"{detections[0]['confidence']:.2f}"]
        else:
            results["YOLO"] = ["No tumor detected", "0.00"]

    elif mode == "CNN (Traditional)":
        with st.spinner("🔍 Running CNN..."):
            result = classify_cnn_image(img)
        results["CNN"] = [result['label'], f"{result['confidence']:.2f}"]

    elif mode == "VGG16":
        with st.spinner("🔍 Running VGG16..."):
            label, conf = classify_vgg16_image(img)
        results["VGG16"] = [label, f"{conf:.2f}"]

    elif mode == "MobileNet":
        with st.spinner("🔍 Running MobileNet..."):
            result = classify_mobilenet_image(img)
        results["MobileNet"] = [result['label'], f"{result['confidence']:.2f}"]

    elif mode == "EfficientNet":
        with st.spinner("🔍 Running EfficientNet..."):
            pred, probs = classify_effnet_image(uploaded_file)
            class_names = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
            results["EfficientNet"] = [class_names[pred], f"{max(probs):.2f}"]

            st.markdown("### 📊 Class Probabilities (EfficientNet)")
            df_probs = pd.DataFrame({"Class": class_names, "Confidence": [round(p, 2) for p in probs]})
            st.table(df_probs)

    elif mode == "Inception (Ensemble)":
        with st.spinner("🔍 Running Inception Ensemble..."):
            result = classify_inception_image(img)
        results["Inception (Ensemble)"] = [result['label'], f"{result['confidence']:.2f}"]

        st.markdown("### 📊 Top-3 Predictions (Inception Ensemble)")
        df_top3 = pd.DataFrame(result["top3"])
        df_top3["p"] = df_top3["p"].round(2)
        st.table(df_top3.rename(columns={"label": "Class", "p": "Confidence"}))
        
    elif mode == "DenseNet169+CBAM":
        with st.spinner("🔍 Running DenseNet169+CBAM..."):
            result = classify_densenet_cbam_image(img)
        results["DenseNet169+CBAM"] = [result['label'], f"{result['confidence']:.2f}"]

        st.markdown("### 📊 Class Probabilities (DenseNet169+CBAM)")
        df_probs = pd.DataFrame(list(result["probs"].items()), columns=["Class", "Confidence"])
        df_probs["Confidence"] = df_probs["Confidence"].round(2)
        st.table(df_probs)
        
    elif mode == "GAN+CNN":
        with st.spinner("🔍 Running GAN+CNN..."):
            result = classify_gan_cnn_image(img)
        results["GAN+CNN"] = [result['label'], f"{result['confidence']:.2f}"]

        st.markdown("### 📊 Class Probabilities (GAN+CNN)")
        df_probs = pd.DataFrame(list(result["probs"].items()), columns=["Class", "Confidence"])
        df_probs["Confidence"] = df_probs["Confidence"].round(2)
        st.table(df_probs)



    # ------------------- Run All Models (with Visualizations) -------------------
    elif mode == "Run All Models":
        st.markdown("## 🚀 Running All Models with Visualizations")

        r1 = classify_image(img)
        results["ResNet+ViT"] = [r1['label'], f"{r1['confidence']:.2f}"]

        detections, plotted_img = detect_image(img)
        results["YOLO"] = [detections[0]['label'], f"{detections[0]['confidence']:.2f}"] if detections else ["No tumor", "0.00"]

        r3 = classify_cnn_image(img)
        results["CNN"] = [r3['label'], f"{r3['confidence']:.2f}"]

        label, conf = classify_vgg16_image(img)
        results["VGG16"] = [label, f"{conf:.2f}"]

        r5 = classify_mobilenet_image(img)
        results["MobileNet"] = [r5['label'], f"{r5['confidence']:.2f}"]

        pred, probs = classify_effnet_image(uploaded_file)
        class_names = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
        results["EfficientNet"] = [class_names[pred], f"{max(probs):.2f}"]
        # DenseNet169+CBAM
        r6 = classify_densenet_cbam_image(img)
        results["DenseNet169+CBAM"] = [r6['label'], f"{r6['confidence']:.2f}"]


        r7 = classify_inception_image(img)
        results["Inception"] = [r7['label'], f"{r7['confidence']:.2f}"]
        
        r_gan = classify_gan_cnn_image(img)
        results["GAN+CNN"] = [r_gan['label'], f"{r_gan['confidence']:.2f}"]


        # ---- Display Results ----
        st.markdown("### 📑 Prediction Results (All Models)")
        df = pd.DataFrame(results, index=["Prediction", "Confidence"]).T
        st.table(df)

        # ---- Confidence Comparison ----
        try:
            confs = [float(v[1]) for v in results.values()]
            models = list(results.keys())
            chart_df = pd.DataFrame({"Model": models, "Confidence": confs}).set_index("Model")
            st.bar_chart(chart_df)
        except:
            st.warning("Some models did not return numeric confidence values.")

    # ------------------- Single Model Results Display -------------------
    if mode != "Run All Models" and results:
        st.markdown("## 📑 Prediction Result")
        df = pd.DataFrame(results, index=["Prediction", "Confidence"]).T
        st.table(df)
