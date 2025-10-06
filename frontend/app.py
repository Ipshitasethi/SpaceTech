import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
from datetime import datetime
import pyttsx3
import threading
from collections import defaultdict
import requests
import base64
import io
import yaml
import glob

# ------------------------ CONFIG ------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")
RESULTS_PATH = os.path.join(BASE_DIR, "runs")
os.makedirs(RESULTS_PATH, exist_ok=True)

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

EXAMPLES_PATH = os.path.abspath(os.path.join(BASE_DIR, CONFIG["paths"]["examples"]))
MODELS_PATH = os.path.abspath(os.path.join(BASE_DIR, CONFIG["paths"]["models"]))

# Backend API URL
BACKEND_URL = "https://spacetech-fj49.onrender.com"

# ------------------------ PAGE CONFIG ------------------------
st.set_page_config(page_title="AstraSight", layout="wide")

# ------------------------ FACT DATA ------------------------
object_facts = {
    "oxygentank": {"what": "Stores breathable oxygen for astronauts.",
                   "use": "Used during EVAs and emergencies to provide life support."},
    "toolbox": {"what": "Contains essential tools for space repairs.",
                "use": "Used by astronauts to fix panels and cables during spacewalks."},
    "fireextinguisher": {"what": "Used to suppress fires in microgravity.",
                         "use": "Activated in the event of fire to protect crew."},
    "default": {"what": "No data available.", "use": "No usage instructions found."}
}

spoken_classes = set()
object_counts = defaultdict(int)

# ------------------------ SPEECH THREADING ------------------------
def speak_threaded(text):
    def run_speech():
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run_speech).start()

def speak_once(label):
    if enable_speech and label not in spoken_classes:
        spoken_classes.add(label)
        speak_threaded(label)

# ------------------------ HELPERS ------------------------
def resize_frame(frame, width=320):
    h, w = frame.shape[:2]
    scale = width / w
    return cv2.resize(frame, (int(w * scale), int(h * scale)))

def danger_status(detected):
    if "fireextinguisher" in detected:
        return "🟢 Safe - Fire extinguisher present."
    elif "toolbox" in detected:
        return "🟡 Caution - Tools in area."
    elif "oxygentank" not in detected:
        return "🔴 Danger - Oxygen supply not found!"
    else:
        return "🟡 Monitor - Ensure equipment visibility."

def predict_backend(image: Image.Image, confidence: float):
    """Send image to FastAPI backend and return annotated image + detections"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()

    response = requests.post(
    f"{BACKEND_URL}/predict",
    files={"file": ("image.jpg", img_bytes, "image/jpeg")},
    data={"confidence": confidence}
)

    response.raise_for_status()
    data = response.json()

    annotated_img_b64 = data["annotated_image"]
    annotated_img = Image.open(io.BytesIO(base64.b64decode(annotated_img_b64)))
    detections = data["detections"]
    return annotated_img, detections

# ------------------------ SIDEBAR ------------------------
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1446776811953-b23d57bd21aa", width=250)
    st.markdown("""### ✨ AstraSight – Key Features
- 🔍 Real-Time Object Detection using YOLOv8
- 🧠 Multi-Class Tracking for mission-critical items
- 🔊 Audio Alerts with confidence scores
- 🖼️ Live Feed & Image Upload Support
- 📊 Object Count & Frame Logging

""")
    st.markdown("### 🤖 Team Name: Visionaries11:11")
    st.markdown("""
    Members:  
    - Ipshita Sethi  
    - Saumya Sudha  
    - Gazal Sindhwani
    - Muskan Goel  
    - Lavanya Sharma 
    - Khushi Ajwani 
    """)
    st.markdown("---")
    confidence = st.slider("🎯 Confidence Threshold", 0.1, 1.0, 0.25, step=0.01)
    enable_speech = st.checkbox("🗣 Enable Voice Feedback", value=True)

# ------------------------ HEADER ------------------------
st.image("https://images.unsplash.com/photo-1581090700227-1e7e54bc59f4", use_container_width=True)
st.markdown("""
    <h1 style='text-align: center; color: #FF4B4B;'>🚀 AstraSight</h1>
    <h4 style='text-align: center; color: gray;'>Real-Time Safety in Zero Gravity</h4>
""", unsafe_allow_html=True)
st.markdown("Detects toolbox, oxygen tank, and fire extinguisher using YOLOv8")

# ------------------------ INPUT MODE ------------------------
option = st.radio("Select input method:", ("Upload Image", "Use Example Image", "Use Webcam"))

# ------------------------ WEBCAM DETECTION ------------------------
if option == "Use Webcam":
    run = st.checkbox("📸 Start Camera")
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    save_button = st.button("💾 Save Current Frame")

    frame_count=0

    while run:
        frame_count += 1
        if frame_count % 2 != 0:  # skip odd frames
            continue
    
        ret, frame = camera.read()
        if not ret:
            st.error("Webcam not accessible.")
            break

        frame = resize_frame(frame)
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        annotated_img, detections = predict_backend(pil_frame, confidence)
        FRAME_WINDOW.image(np.array(annotated_img))

        detected_labels = [d["label"] for d in detections]
        for d in detections:
            speak_once(f'{d["label"]} detected with {d["confidence"]:.0%} confidence')

        if save_button:
            filename = os.path.join(RESULTS_PATH,f"saved_frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            annotated_img.save(filename)
            st.success(f"Frame saved as {filename}")

        if detected_labels:
            st.markdown(f"### 🛡 Status: {danger_status(detected_labels)}")

# ------------------------ IMAGE UPLOAD DETECTION ------------------------
elif option == "Upload Image":
    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        annotated_img, detections = predict_backend(image, confidence)
        st.image(annotated_img, caption="🛰 Detection Results", use_container_width=True)

        detected_labels = [d["label"] for d in detections]
        for d in detections:
            speak_once(f'{d["label"]} detected with {d["confidence"]:.0%} confidence')

        st.markdown("### 📋 Detected Objects")
        for label in detected_labels:
            obj = object_facts.get(label.lower(), object_facts["default"])
            st.markdown(f"<small>🧠 What it is: {obj['what']}</small>", unsafe_allow_html=True)
            st.markdown(f"<small>🛠 How it’s used: {obj['use']}</small>", unsafe_allow_html=True)

        st.markdown(f"### 🛡 Status: {danger_status(detected_labels)}")

# ------------------------ EXAMPLE IMAGE ------------------------
elif option == "Use Example Image":
    example_files = glob.glob(os.path.join(EXAMPLES_PATH, "*.*"))
    example_names = [os.path.basename(f) for f in example_files]
    selected_example = st.selectbox("Select an example image", example_names)

    if selected_example:
        example_path = os.path.join(EXAMPLES_PATH, selected_example)
        image = Image.open(example_path).convert("RGB")
        annotated_img, detections = predict_backend(image, confidence)
        st.image(annotated_img, caption=f"🛰 Detection Results on {selected_example}", use_container_width=True)

        detected_labels = [d["label"] for d in detections]
        for d in detections:
            speak_once(f'{d["label"]} detected with {d["confidence"]:.0%} confidence')

        st.markdown("### 📋 Detected Objects")
        for label in detected_labels:
            obj = object_facts.get(label.lower(), object_facts["default"])
            st.markdown(f"<small>🧠 What it is: {obj['what']}</small>", unsafe_allow_html=True)
            st.markdown(f"<small>🛠 How it’s used: {obj['use']}</small>", unsafe_allow_html=True)

        st.markdown(f"### 🛡 Status: {danger_status(detected_labels)}")

# ------------------------ FOOTER ------------------------
st.markdown("---")
st.markdown("🛠 Developed for SIH 2025")
