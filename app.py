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
import gdown
import matplotlib.pyplot as plt

from classification import StrokeClassifier, StrokeTypeClassifier, predict_class
from segmentation_detection import UNet, extract_clots_from_mask

# Set page config for a wider layout and custom title
st.set_page_config(page_title="Brain CT Stroke & Clot Detection", page_icon="🧠", layout="wide")

# Custom CSS for a better aesthetic
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main .block-container{
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #1e2127;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #333;
    }
    .metric-value-high {
        color: #ff4b4b;
        font-size: 24px;
        font-weight: bold;
    }
    .metric-value-medium {
        color: #ffa421;
        font-size: 24px;
        font-weight: bold;
    }
    .metric-value-low {
        color: #008f51;
        font-size: 24px;
        font-weight: bold;
    }
    .metric-value-neutral {
        color: #ffffff;
        font-size: 24px;
        font-weight: bold;
    }
    .metric-label {
        color: #a3a8b8;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Real CNN Inference Pipelines
# -----------------------------------------------------------------------------

# 1. Model Caching (Loads weights once in Streamlit)
@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -------------------------------
    # DOWNLOAD MODEL FILES FROM DRIVE
    # -------------------------------
    def download_file(file_id, output):
        if not os.path.exists(output):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output, quiet=False)

    download_file("14IgpgBioDyohj8VFbELbTAuRxOsRx0rf", "stroke_classifier_weights.pth")
    download_file("1hGEHj_pcsAzLmUWPUQg0LqpRThnUjB61", "stroke_type_weights.pth")
    download_file("1yGySxyxfLMnjWtzO-TwAV9z9gldmZMrc", "unet_weights.pth")

    # -------------------------------
    # LOAD MODELS (UNCHANGED)
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

# 2. Image Preprocessing Pipelines
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
    """Predicts if the image is Normal or has a Stroke using ResNet."""
    image_t = classification_transforms(image.convert("RGB")).unsqueeze(0).to(device)
    classes = get_classes_for_model("stroke_classifier_weights.pth", ["Normal", "Stroke"])
    return predict_class(model_stroke, image_t, classes)

def predict_stroke_type(image: Image.Image) -> str:
    """Predicts if the stroke is Ischemic or Hemorrhagic using ResNet."""
    image_t = classification_transforms(image.convert("RGB")).unsqueeze(0).to(device)
    classes = get_classes_for_model("stroke_type_weights.pth", ["Hemorrhagic", "Ischemic"])
    return predict_class(model_type, image_t, classes)

def detect_clots_and_lesion(image: Image.Image, conf_threshold: float = 0.5):
    """Outputs number of clots, area, and annotated image using U-Net."""
    img_rgb = image.convert("RGB")
    original_size = img_rgb.size # (width, height)
    
    # 1. Forward Pass U-Net
    image_t = segmentation_transforms(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        mask_pred = model_unet(image_t).squeeze().cpu().numpy() # [256, 256] probability mask
    
    # 1.5 Create a Brain Mask to ignore the black background frame
    img_resized_gray = cv2.cvtColor(np.array(img_rgb.resize((256, 256))), cv2.COLOR_RGB2GRAY)
    _, brain_mask = cv2.threshold(img_resized_gray, 15, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5),np.uint8)
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)
    mask_pred[brain_mask == 0] = 0.0
    
    # 2. Process Mask into contours & metrics
    num_clots, clot_areas_scaled, contours, binary_mask = extract_clots_from_mask(mask_pred, threshold=conf_threshold)
    
    scale_factor_x = original_size[0] / 256.0
    scale_factor_y = original_size[1] / 256.0
    
    estimated_brain_area = int(np.count_nonzero(brain_mask) * scale_factor_x * scale_factor_y)
    
    lesion_area_texts = []
    total_lesion_area = 0
    actual_areas_list = []
    
    for i, area_scaled in enumerate(clot_areas_scaled):
        actual_area = int(area_scaled * scale_factor_x * scale_factor_y)
        lesion_area_texts.append(f"Clot {i+1}: {actual_area} px²")
        total_lesion_area += actual_area
        actual_areas_list.append(actual_area)
        
    if not lesion_area_texts:
        lesion_area_str = "0 px²"
    else:
        lesion_area_str = "<br>".join(lesion_area_texts)
    
    # 3. Draw Annotations on Original Image
    img_array = np.array(img_rgb)
    for cnt in contours:
        cnt[:, 0, 0] = (cnt[:, 0, 0] * scale_factor_x).astype(int)
        cnt[:, 0, 1] = (cnt[:, 0, 1] * scale_factor_y).astype(int)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_array, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img_array, 'Lesion', (x, max(10, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.drawContours(img_array, [cnt], -1, (0, 0, 255), 1)

    annotated_image = Image.fromarray(img_array)
    return num_clots, total_lesion_area, lesion_area_str, actual_areas_list, estimated_brain_area, annotated_image

def calculate_risk(stroke_pred: str, stroke_type: str, num_clots: int, total_area: int) -> str:
    """Calculates risk level based on logic in src/risk_engine.py."""
    if stroke_pred == "Normal":
        return "Low"
    
    if stroke_type == "Hemorrhagic" or num_clots > 1 or total_area > 2000:
         return "High"
    elif stroke_type == "Ischemic" and num_clots > 0:
        return "Medium"
    else:
        return "Low"


# -----------------------------------------------------------------------------
# NEW: Individual Patient Report Generator
# -----------------------------------------------------------------------------
def show_patient_report(stroke_pred, stroke_type, num_clots, total_lesion_area, actual_areas_list, brain_area, annotated_image, risk_level):
    st.markdown("<h3 style='text-align: center;'>📋 Comprehensive Patient Data Analysis Report</h3>", unsafe_allow_html=True)
    st.markdown("<hr/>", unsafe_allow_html=True)
    
    # 1. Summary details output (Like you asked!)
    r_color = {"High": "#ff4b4b", "Medium": "#ffa421", "Low": "#008f51"}.get(risk_level, "#ffffff")
    st.markdown(f"""
    1. **Anomaly Condition:** {stroke_pred}
    2. **Classified Type:** {stroke_type}
    3. **Blood Clots Logged:** {num_clots} isolated region(s)
    4. **Sum of Damaged Tissue:** {total_lesion_area} px²
    5. **Automated Risk Assessment:** <strong style="color:{r_color};">{risk_level} RISK LEVEL</strong>
    """, unsafe_allow_html=True)
    
    st.markdown("<br/>", unsafe_allow_html=True)
    
    # 2. Graphs
    colA, colB = st.columns(2)
    with colA:
        st.markdown("<h5 style='text-align: center'>Individual Clot Sizes (Bar Graph)</h5>", unsafe_allow_html=True)
        if num_clots > 0 and actual_areas_list:
            df_clots = pd.DataFrame({
                "Clot": [f"Clot {i+1}" for i in range(len(actual_areas_list))],
                "Area in Pixels": actual_areas_list
            }).set_index("Clot")
            st.bar_chart(df_clots)
        else:
            st.info("No lesions detected. Graph unavailable.")

    with colB:
        st.markdown("<h5 style='text-align: center'>Overall Risk Severity Impact (Pie Chart)</h5>", unsafe_allow_html=True)
        
        # Calculate visual severity impact
        if risk_level == "High":
             risk_score = min(98.0, 85.0 + (num_clots * 2.5))
        elif risk_level == "Medium":
             risk_score = min(75.0, 50.0 + (num_clots * 5.0))
        else:
             risk_score = 12.0
             
        safe_score = 100.0 - risk_score
        
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        fig.patch.set_alpha(0.0)
        ax.pie(
            [safe_score, risk_score], 
            labels=["Normal/Safe Factor", "Critical Risk Factor"], 
            colors=["#008f51", "#ff4b4b"], 
            autopct='%1.1f%%', 
            startangle=140,
            textprops={'color':"white", 'weight':'bold', 'fontsize':10}
        )
        ax.axis('equal')
        st.pyplot(fig)

    # 3. Segmentation Map Output
    st.markdown("<br/><h5 style='text-align: center'>Final Lesion Segmentation Map</h5>", unsafe_allow_html=True)
    col_img1, col_img2, col_img3 = st.columns([1,2,1])
    with col_img2:
        st.image(annotated_image, caption="AI-Identified Lesions visually indicated.", use_container_width=True)


# -----------------------------------------------------------------------------
# Main Application UI
# -----------------------------------------------------------------------------
def main():
    st.title("🧠 Brain CT Stroke & Clot Detection System")
    st.markdown("Upload a patient's CT Scan image to analyze for stokes and blood clots.")

    st.sidebar.header("Controls")
    uploaded_file = st.sidebar.file_uploader("Upload CT Scan (.jpg, .png, .dcm)", type=["jpg", "jpeg", "png"])

    # Model Settings placeholder
    st.sidebar.subheader("Model Settings")
    confidence_threshold = st.sidebar.slider("Detection Confidence", min_value=0.1, max_value=1.0, value=0.5, step=0.05)

    if uploaded_file is not None:
        # Load Image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.subheader("Original CT Scan")
            st.image(image, use_container_width=True, caption="Uploaded Image")
            
        with col2:
            st.subheader("Analysis Results")
            
            with st.spinner('Analyzing CT Scan...'):
                try:
                    stroke_pred = predict_stroke(image)
                    
                    stroke_type = "N/A"
                    if stroke_pred == "Stroke":
                        stroke_type = predict_stroke_type(image)
                        
                    # EXTRACT EVERYTHING NECESSARY FOR THE REPORT
                    num_clots, total_lesion_area, lesion_area_str, actual_areas_list, brain_area, annotated_image = detect_clots_and_lesion(image, conf_threshold=confidence_threshold)
                    
                    risk_level = calculate_risk(stroke_pred, stroke_type, num_clots, total_lesion_area)
                    
                except Exception as e:
                     st.error(f"Error processing image: {e}")
                     return

            # Row 1: Primary Diagnoses
            st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                color_class = "metric-value-high" if stroke_pred == "Stroke" else "metric-value-low"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Stroke Prediction</div>
                    <div class="{color_class}">{stroke_pred}</div>
                </div>
                """, unsafe_allow_html=True)
            with m_col2:
                color_class = "metric-value-high" if stroke_type == "Hemorrhagic" else ("metric-value-medium" if stroke_type == "Ischemic" else "metric-value-neutral")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Stroke Type</div>
                    <div class="{color_class}">{stroke_type}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height: 15px'></div>", unsafe_allow_html=True)
            
            # Row 2: Clot & Lesion Details
            m_col3, m_col4 = st.columns(2)
            with m_col3:
                 color_class = "metric-value-high" if num_clots > 0 else "metric-value-low"
                 st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Blood Clots Detected</div>
                    <div class="{color_class}">{num_clots}</div>
                </div>
                """, unsafe_allow_html=True)
            with m_col4:
                color_class = "metric-value-medium" if lesion_area_str != "0 px²" else "metric-value-low"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Lesion Damaged Area</div>
                    <div class="{color_class}">{lesion_area_str}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height: 15px'></div>", unsafe_allow_html=True)
            
            # Row 3: Risk Level
            r_color = {"High": "metric-value-high", "Medium": "metric-value-medium", "Low": "metric-value-low"}.get(risk_level, "metric-value-neutral")
            st.markdown(f"""
            <div class="metric-card" style="border: 2px solid {'#ff4b4b' if risk_level=='High' else ('#ffa421' if risk_level=='Medium' else '#008f51')};">
                <div class="metric-label">Risk Prediction Level</div>
                <div class="{r_color}" style="font-size: 32px;">{risk_level} RISK</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Lesion Segmentation Map")
        st.image(annotated_image, use_container_width=True, caption="Detected Clots Component & Lesion Area")

        # ---------------------------------------
        # GENERATE REPORT BUTTON 
        # ---------------------------------------
        st.markdown("<div style='height: 30px'></div>", unsafe_allow_html=True)
        with st.expander("📊 CLICK HERE TO GENERATE FINAL PATIENT REPORT"):
             show_patient_report(stroke_pred, stroke_type, num_clots, total_lesion_area, actual_areas_list, brain_area, annotated_image, risk_level)

    else:
        st.info("Please upload a CT scan image using the sidebar to begin analysis.")
        
        st.markdown("""
        ### Instructions
        1. Select a patient's CT scan image from your local machine.
        2. The system will automatically process the image through 3 multi-stage Neural Networks.
        3. Review the AI-generated results including:
           - **Stroke Prediction:** Determines presence of anomalies.
           - **Stroke Type:** Differentiates between Ischemic and Hemorrhagic.
           - **Clot Detection & Lesion Area:** Maps the extent of brain damage.
           - **Risk Engine:** Assigns a severity score.
        """)

if __name__ == "__main__":
    main()
