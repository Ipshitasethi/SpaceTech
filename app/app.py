import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
from datetime import datetime
import pyttsx3
import threading
from collections import defaultdict

import yaml


# Get absolute path to config.yaml no matter where script is run from
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")
RESULTS_PATH = os.path.join(BASE_DIR, "runs")

print("BASE_DIR:", BASE_DIR)

# Ensure results directory exists
os.makedirs(RESULTS_PATH, exist_ok=True)



# Load configuration file
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

EXAMPLES_PATH = os.path.abspath(os.path.join(BASE_DIR, CONFIG["paths"]["examples"]))
MODELS_PATH = os.path.abspath(os.path.join(BASE_DIR, CONFIG["paths"]["models"]))





# --- PAGE CONFIG ---
st.set_page_config(page_title="Falcon Eye", layout="wide")

# --- FACT DATA ---
object_facts = {
    "oxygentank": {
        "what": "Stores breathable oxygen for astronauts.",
        "use": "Used during EVAs and emergencies to provide life support."
    },
    "toolbox": {
        "what": "Contains essential tools for space repairs.",
        "use": "Used by astronauts to fix panels and cables during spacewalks."
    },
    "fireextinguisher": {
        "what": "Used to suppress fires in microgravity.",
        "use": "Activated in the event of fire to protect crew."
    },
    "default": {
        "what": "No data available.",
        "use": "No usage instructions found."
    }
}

# --- INIT STATE ---
spoken_classes = set()
object_counts = defaultdict(int)

# --- SPEECH THREADING ---
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

# --- RESIZE IMAGE ---
def resize_frame(frame, width=640):
    h, w = frame.shape[:2]
    scale = width / w
    return cv2.resize(frame, (int(w * scale), int(h * scale)))

# --- STATUS DISPLAY ---
def danger_status(detected):
    if "fireextinguisher" in detected:
        return "🟢 Safe - Fire extinguisher present."
    elif "toolbox" in detected:
        return "🟡 Caution - Tools in area."
    elif "oxygentank" not in detected:
        return "🔴 Danger - Oxygen supply not found!"
    else:
        return "🟡 Monitor - Ensure equipment visibility."

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1446776811953-b23d57bd21aa", width=250)
    st.markdown("### 🚀 Built for Microsoft + Duality AI")
    st.markdown("### 🤖 Team: Snapcode")
    st.markdown("""
    Members:  
    - Saumya Sudha  
    - Ipshita Sethi  
    - Muskan Goel  
    - Lavanya Sharma  
    """)
    st.markdown("---")
    model_files = [f for f in os.listdir(MODELS_PATH) if f.endswith((".pt", ".onnx"))]
    model_name = st.selectbox("📦 Select YOLO Model", model_files)
    confidence = st.slider("🎯 Confidence Threshold", 0.1, 1.0, 0.25, step=0.01)
    enable_speech = st.checkbox("🗣 Enable Voice Feedback", value=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model(path):
    return YOLO(path)

model = load_model(os.path.join(MODELS_PATH, model_name))


# --- HEADER & BANNER ---
st.image("https://images.unsplash.com/photo-1581090700227-1e7e54bc59f4", use_container_width=True)
st.markdown("""
    <h1 style='text-align: center; color: #FF4B4B;'>🦅 Falcon Eye</h1>
    <h4 style='text-align: center; color: gray;'>Space Station Object Detector</h4>
""", unsafe_allow_html=True)

st.markdown("Detects toolbox, oxygen tank, and fire extinguisher using YOLOv8")

# --- INPUT MODE ---
option = st.radio("Select input method:", ("Upload Image", "Use Example Image", "Use Webcam"))

# --- WEBCAM DETECTION ---
if option == "Use Webcam":
    run = st.checkbox("📸 Start Camera")
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    save_button = st.button("💾 Save Current Frame")

    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("Webcam not accessible.")
            break

        frame = resize_frame(frame)
        results = model.predict(frame, conf=confidence, iou=0.5)
        boxes = results[0].boxes

        detected_labels = []

        if len(boxes) == 0:
            spoken_classes.clear()

        for box in boxes:
            coords = box.xyxy[0].tolist()
            class_id = int(box.cls[0])
            conf_score = box.conf[0].item()
            class_name = model.names[class_id].lower()
            label_text = f"{class_name} ({conf_score:.2f})"
            speak_label = f"{class_name} detected with {conf_score:.0%} confidence"

            detected_labels.append(class_name)
            object_counts[class_name] += 1
            speak_once(speak_label)

            cv2.rectangle(frame, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (int(coords[0]), int(coords[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if save_button:
            filename = os.path.join(RESULTS_PATH,f"saved_frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(filename, frame)
            st.success(f"Frame saved as {filename}")

        if detected_labels:
            st.markdown(f"### 🛡 Status: {danger_status(detected_labels)}")

# --- IMAGE UPLOAD DETECTION ---
elif option == "Upload Image":
    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = resize_frame(np.array(image))

        with st.spinner("🔍 Running detection..."):
            results = model.predict(img_array, conf=confidence, iou=0.5)
            boxes = results[0].boxes

        detected_labels = []

        if len(boxes) == 0:
            st.warning("❗ No objects detected.")
            spoken_classes.clear()

        for box in boxes:
            coords = box.xyxy[0].tolist()
            class_id = int(box.cls[0])
            conf_score = box.conf[0].item()
            class_name = model.names[class_id].lower()
            label_text = f"{class_name} ({conf_score:.2f})"
            speak_label = f"{class_name} detected with {conf_score:.0%} confidence"

            detected_labels.append(class_name)
            object_counts[class_name] += 1
            speak_once(speak_label)

            cv2.rectangle(img_array, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), (0, 255, 0), 2)
            cv2.putText(img_array, label_text, (int(coords[0]), int(coords[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        st.image(img_array, caption="🛰 Detection Results", use_container_width=True)

        if st.button("💾 Save Output Image"):
            output_filename = os.path.join(RESULTS_PATH, f"detected_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(output_filename, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
            st.success(f"Image saved as {output_filename}")

        st.markdown("### 📋 Detected Objects")

        for label in detected_labels:
            conf_list = [box.conf[0].item() for box in boxes if model.names[int(box.cls[0])].lower() == label]
            if conf_list:
                st.write(f"- {label.title()} — Confidence: {conf_list[0]:.2f}")
            obj = object_facts.get(label.lower(), object_facts["default"])
            st.markdown(f"<small>🧠 What it is: {obj['what']}</small>", unsafe_allow_html=True)
            st.markdown(f"<small>🛠 How it’s used: {obj['use']}</small>", unsafe_allow_html=True)

        st.markdown(f"### 🛡 Status: {danger_status(detected_labels)}")


elif option == "Use Example Image":
    import glob
    
    # Load example image file paths
    example_files = glob.glob(os.path.join(EXAMPLES_PATH, "*.*"))  # All files in example folder
    
    # Let user select from example images by filename
    example_names = [os.path.basename(f) for f in example_files]
    
    selected_example = st.selectbox("Select an example image", example_names)
    
    if selected_example:
        example_path = os.path.join(EXAMPLES_PATH, selected_example)
        image = Image.open(example_path).convert("RGB")
        img_array = resize_frame(np.array(image))
        
        with st.spinner("🔍 Running detection on example image..."):
            results = model.predict(img_array, conf=confidence, iou=0.5)
            boxes = results[0].boxes
        
        detected_labels = []
        if len(boxes) == 0:
            st.warning("❗ No objects detected.")
            spoken_classes.clear()

        for box in boxes:
            coords = box.xyxy[0].tolist()
            class_id = int(box.cls[0])
            conf_score = box.conf[0].item()
            class_name = model.names[class_id].lower()
            label_text = f"{class_name} ({conf_score:.2f})"
            speak_label = f"{class_name} detected with {conf_score:.0%} confidence"

            detected_labels.append(class_name)
            object_counts[class_name] += 1
            speak_once(speak_label)

            cv2.rectangle(img_array, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), (0, 255, 0), 2)
            cv2.putText(img_array, label_text, (int(coords[0]), int(coords[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        st.image(img_array, caption=f"🛰 Detection Results on {selected_example}", use_container_width=True)

        if st.button("💾 Save Output Image"):
            output_filename = os.path.join(RESULTS_PATH, f"detected_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(output_filename, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
            st.success(f"Image saved as {output_filename}")

        st.markdown("### 📋 Detected Objects")
        for label in detected_labels:
            conf_list = [box.conf[0].item() for box in boxes if model.names[int(box.cls[0])].lower() == label]
            if conf_list:
                st.write(f"- {label.title()} — Confidence: {conf_list[0]:.2f}")
            obj = object_facts.get(label.lower(), object_facts["default"])
            st.markdown(f"<small>🧠 What it is: {obj['what']}</small>", unsafe_allow_html=True)
            st.markdown(f"<small>🛠 How it’s used: {obj['use']}</small>", unsafe_allow_html=True)

        st.markdown(f"### 🛡 Status: {danger_status(detected_labels)}")



# --- FOOTER ---
st.markdown("---")
st.markdown("🛠 Developed for BuildWithIndia2.0 — Microsoft + Duality AI Hackathon")
