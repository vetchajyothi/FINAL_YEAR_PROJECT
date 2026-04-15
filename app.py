import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import time
import os
import io
import torch
from torchvision import transforms
import cv2
import gdown   # ✅ added

from src.models.classification import StrokeClassifier, StrokeTypeClassifier, predict_class
from src.models.segmentation_detection import UNet, extract_clots_from_mask

# Set page config for a wider layout and custom title
st.set_page_config(page_title="Brain CT Stroke & Clot Detection", page_icon="🧠", layout="wide")

# Custom CSS (UNCHANGED)
st.markdown("""
<style>
    .reportview-container { background: #0e1117; }
    .main .block-container{ padding-top: 2rem; }
    .metric-card {
        background-color: #1e2127;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #333;
    }
    .metric-value-high { color: #ff4b4b; font-size: 24px; font-weight: bold; }
    .metric-value-medium { color: #ffa421; font-size: 24px; font-weight: bold; }
    .metric-value-low { color: #008f51; font-size: 24px; font-weight: bold; }
    .metric-value-neutral { color: #ffffff; font-size: 24px; font-weight: bold; }
    .metric-label {
        color: #a3a8b8;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 🔥 DOWNLOAD FUNCTION (NEW)
# ------------------------------------------------------------------
def download_file(url, output):
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

# ------------------------------------------------------------------
# MODEL LOADING (ONLY THIS PART MODIFIED)
# ------------------------------------------------------------------
@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 🔥 DOWNLOAD FROM GOOGLE DRIVE
    download_file("https://drive.google.com/uc?id=14IgpgBioDyohj8VFbELbTAuRxOsRx0rf",
                  "stroke_classifier_weights.pth")

    download_file("https://drive.google.com/uc?id=1hGEHj_pcsAzLmUWPUQg0LqpRThnUjB61",
                  "stroke_type_weights.pth")

    download_file("https://drive.google.com/uc?id=1yGySxyxfLMnjWtzO-TwAV9z9gldmZMrc",
                  "unet_weights.pth")

    # -------------------------------
    # LOAD MODELS (UNCHANGED LOGIC)
    # -------------------------------
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

# ------------------------------------------------------------------
# REST OF YOUR CODE (100% SAME)
# ------------------------------------------------------------------

classification_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

segmentation_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

import json
def get_classes_for_model(weights_path, default_classes):
    json_path = weights_path + "_classes.json"
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    return default_classes

def predict_stroke(image: Image.Image) -> str:
    image_t = classification_transforms(image.convert("RGB")).unsqueeze(0).to(device)
    classes = get_classes_for_model("stroke_classifier_weights.pth", ["Normal", "Stroke"])
    return predict_class(model_stroke, image_t, classes)

def predict_stroke_type(image: Image.Image) -> str:
    image_t = classification_transforms(image.convert("RGB")).unsqueeze(0).to(device)
    classes = get_classes_for_model("stroke_type_weights.pth", ["Hemorrhagic", "Ischemic"])
    return predict_class(model_type, image_t, classes)

def detect_clots_and_lesion(image: Image.Image, conf_threshold: float = 0.5):
    img_rgb = image.convert("RGB")
    original_size = img_rgb.size

    image_t = segmentation_transforms(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        mask_pred = model_unet(image_t).squeeze().cpu().numpy()

    img_resized_gray = cv2.cvtColor(np.array(img_rgb.resize((256, 256))), cv2.COLOR_RGB2GRAY)
    _, brain_mask = cv2.threshold(img_resized_gray, 15, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5,5),np.uint8)
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)

    mask_pred[brain_mask == 0] = 0.0

    num_clots, clot_areas_scaled, contours, binary_mask = extract_clots_from_mask(mask_pred, threshold=conf_threshold)

    scale_factor_x = original_size[0] / 256.0
    scale_factor_y = original_size[1] / 256.0

    total_lesion_area = 0
    for area_scaled in clot_areas_scaled:
        total_lesion_area += int(area_scaled * scale_factor_x * scale_factor_y)

    img_array = np.array(img_rgb)
    for cnt in contours:
        cnt[:, 0, 0] = (cnt[:, 0, 0] * scale_factor_x).astype(int)
        cnt[:, 0, 1] = (cnt[:, 0, 1] * scale_factor_y).astype(int)

        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_array, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.drawContours(img_array, [cnt], -1, (0, 0, 255), 1)

    return num_clots, total_lesion_area, Image.fromarray(img_array)

def calculate_risk(stroke_pred, stroke_type, num_clots, total_area):
    if stroke_pred == "Normal":
        return "Low"
    if stroke_type == "Hemorrhagic" or num_clots > 1 or total_area > 2000:
        return "High"
    elif stroke_type == "Ischemic" and num_clots > 0:
        return "Medium"
    else:
        return "Low"

def main():
    st.title("🧠 Brain CT Stroke & Clot Detection System")
    st.sidebar.header("Controls")

    uploaded_file = st.sidebar.file_uploader("Upload CT Scan", type=["jpg","png","jpeg"])
    confidence_threshold = st.sidebar.slider("Detection Confidence",0.1,1.0,0.5)

    if uploaded_file:
        image = Image.open(uploaded_file)

        col1,col2 = st.columns([1,1.2])

        with col1:
            st.image(image,caption="Uploaded")

        with col2:
            stroke_pred = predict_stroke(image)
            stroke_type = predict_stroke_type(image) if stroke_pred=="Stroke" else "N/A"
            num_clots,total_area,annotated = detect_clots_and_lesion(image,confidence_threshold)
            risk = calculate_risk(stroke_pred,stroke_type,num_clots,total_area)

            st.write("Stroke:",stroke_pred)
            st.write("Type:",stroke_type)
            st.write("Clots:",num_clots)
            st.write("Risk:",risk)

        st.image(annotated)

if __name__ == "__main__":
    main()