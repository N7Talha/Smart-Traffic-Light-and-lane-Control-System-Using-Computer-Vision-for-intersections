import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os

# -------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = "D:/Work/Atomcamp/Final Project/Final Project code/runs/detect/udacity_yolo11x/weights/best.pt"
    return YOLO(model_path)

model = load_model()

# -------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------
st.title("🚦 Smart Traffic System — YOLO Inference")
st.write("Upload an image of an intersection, and the model will detect vehicles, people, and traffic lights.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert to PIL Image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Temporary file for YOLO
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    st.write("🔍 Running inference...")
    
    # -------------------------------------------------------
    # YOLO INFERENCE
    # -------------------------------------------------------
    results = model.predict(temp_path, imgsz=640)

    # Get rendered image from YOLO
    result_img = results[0].plot()

    # Show output
    st.image(result_img, caption="Detection Result", use_column_width=True)

    # -------------------------------------------------------
    # OPTIONAL: Show detections as text
    # -------------------------------------------------------
    st.subheader(" Detected Objects")
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        st.write(f"• {model.names[cls]} — {conf:.2f}")

    # Remove temp file
    os.remove(temp_path)
