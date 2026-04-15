```python
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import torch
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
import gdown

from classification import StrokeClassifier, StrokeTypeClassifier, predict_class
from segmentation_detection import UNet, extract_clots_from_mask

# Page config
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

    # Load Stroke Classifier
    model_stroke = StrokeClassifier(num_classes=2).to(device)
    model_stroke.load_state_dict(torch.load("stroke_classifier_weights.pth", map_location=device))
    model_stroke.eval()

    # Load Stroke Type Classifier
    model_type = StrokeTypeClassifier(num_classes=2).to(device)
    model_type.load_state_dict(torch.load("stroke_type_weights.pth", map_location=device))
    model_type.eval()

    # Load U-Net
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
# PREDICTION FUNCTIONS
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

# -------------------------------
# UI
# -------------------------------
def main():
    st.title("🧠 Brain CT Stroke Detection")

    file = st.file_uploader("Upload CT Scan", type=["jpg", "png", "jpeg"])

    if file:
        image = Image.open(file).convert("RGB")

        st.image(image, caption="Uploaded Image")

        if st.button("Analyze"):
            stroke = predict_stroke(image)

            if stroke == "Stroke":
                stroke_type = predict_stroke_type(image)
            else:
                stroke_type = "N/A"

            num_clots, area, annotated = detect_clots_and_lesion(image)

            st.subheader("Results")
            st.write("Stroke:", stroke)
            st.write("Type:", stroke_type)
            st.write("Clots:", num_clots)
            st.write("Lesion Area:", area)

            st.image(annotated, caption="Detected Areas")


if __name__ == "__main__":
    main()
```
