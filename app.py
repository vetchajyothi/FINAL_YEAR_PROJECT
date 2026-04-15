import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import time
import os
import io
import torch
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
import gdown

from classification import StrokeClassifier, StrokeTypeClassifier, predict_class
from segmentation_detection import UNet, extract_clots_from_mask

st.set_page_config(page_title="Brain CT Stroke & Clot Detection", page_icon="🧠", layout="wide")

# -------------------------------
# DOWNLOAD MODEL FILES
# -------------------------------
def download_file(url, output):
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

# -------------------------------
# LOAD MODELS (FIXED)
# -------------------------------
@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Download weights
    download_file("https://drive.google.com/uc?id=14IgpgBioDyohj8VFbELbTAuRxOsRx0rf",
                  "stroke_classifier_weights.pth")

    download_file("https://drive.google.com/uc?id=1hGEHj_pcsAzLmUWPUQg0LqpRThnUjB61",
                  "stroke_type_weights.pth")

    download_file("https://drive.google.com/uc?id=1yGySxyxfLMnjWtzO-TwAV9z9gldmZMrc",
                  "unet_weights.pth")

    # Load models
    model_stroke = StrokeClassifier(num_classes=2).to(device)
    model_stroke.load_state_dict(torch.load("stroke_classifier_weights.pth", map_location=device))
    model_stroke.eval()

    model_type = StrokeTypeClassifier(num_classes=2).to(device)
    model_type.load_state_dict(torch.load("stroke_type_weights.pth", map_location=device))
    model_type.eval()

    model_unet = UNet(n_channels=3, n_classes=1).to(device)
    model_unet.load_state_dict(torch.load("unet_weights.pth", map_location=device))
    model_unet.eval()

    return model_stroke, model_type, model_unet, device


model_stroke, model_type, model_unet, device = load_models()

# -------------------------------
# TRANSFORMS
# -------------------------------
classification_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

segmentation_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# -------------------------------
# FUNCTIONS
# -------------------------------
def predict_stroke(image):
    image_t = classification_transforms(image).unsqueeze(0).to(device)
    return predict_class(model_stroke, image_t, ["Normal", "Stroke"])

def predict_stroke_type(image):
    image_t = classification_transforms(image).unsqueeze(0).to(device)
    return predict_class(model_type, image_t, ["Hemorrhagic", "Ischemic"])

def detect_clots_and_lesion(image):
    img = segmentation_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        mask = model_unet(img).squeeze().cpu().numpy()

    num_clots, areas, contours, _ = extract_clots_from_mask(mask)
    total_area = sum(areas)

    img_array = np.array(image)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_array, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return num_clots, total_area, Image.fromarray(img_array)

def calculate_risk(stroke, stroke_type, num_clots):
    if stroke == "Normal":
        return "Low"
    if stroke_type == "Hemorrhagic" or num_clots > 1:
        return "High"
    return "Medium"

# -------------------------------
# UI
# -------------------------------
def main():
    st.title("🧠 Brain CT Stroke & Clot Detection System")

    st.sidebar.header("Controls")
    file = st.sidebar.file_uploader("Upload CT Scan", type=["jpg", "png", "jpeg"])

    confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.5)

    if file:
        image = Image.open(file).convert("RGB")

        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.subheader("Original CT Scan")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Analysis Results")

            with st.spinner("Analyzing..."):
                stroke = predict_stroke(image)

                if stroke == "Stroke":
                    stroke_type = predict_stroke_type(image)
                else:
                    stroke_type = "N/A"

                num_clots, area, annotated = detect_clots_and_lesion(image)
                risk = calculate_risk(stroke, stroke_type, num_clots)

            st.metric("Stroke Prediction", stroke)
            st.metric("Stroke Type", stroke_type)
            st.metric("Clots Detected", num_clots)
            st.metric("Lesion Area", f"{area} px²")
            st.metric("Risk Level", risk)

        st.image(annotated, caption="Detected Lesions")

    else:
        st.info("Please upload a CT scan image to begin analysis.")

if __name__ == "__main__":
    main()