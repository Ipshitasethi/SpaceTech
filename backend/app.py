# backend/app.py
import os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from functools import lru_cache
from datetime import datetime
import base64
import io
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

app= FastAPI()
@app.get("/")
def root():
    return {"message": "API is live! Welcome to YOLO Detection Service."}


# Basic config
DEFAULT_MODEL = "best.pt"  # you can change this to alternate.pt or fire_safety.pt
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join("models", DEFAULT_MODEL))
CONFIDENCE_DEFAULT = float(os.environ.get("CONFIDENCE_DEFAULT", 0.25))

app = FastAPI(title="Falcon Eye Model Server")

# Allow requests from your Streamlit origin (for dev keep '*' ; tighten in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------------
# Ensure ultralytics DetectionModel can be deserialized safely (if needed)
@lru_cache()
def get_model():
    # If your environment needs the safe globals trick (you used earlier), keep it:
    try:
        import torch
        from ultralytics.nn.tasks import DetectionModel
        torch.serialization.add_safe_globals([DetectionModel])
    except Exception:
        pass
    model = YOLO(MODEL_PATH)
    return model

# helper: decode bytes -> BGR numpy image
def read_imagefile(file_bytes: bytes):
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR
    return img

# helper: run inference and return detections + annotated image (base64)
def run_inference_and_annotate(img_bgr, conf) -> (List[dict], str):
    model = get_model()
    # Ultralyics accepts BGR in your current code — we'll pass through what you used earlier
    results = model.predict(img_bgr, conf=conf, iou=0.5)  # returns a Results object list
    detections = []
    for box in results[0].boxes:
        coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
        class_id = int(box.cls[0])
        conf_score = float(box.conf[0])
        label = model.names[class_id].lower()
        detections.append({
            "label": label,
            "confidence": conf_score,
            "bbox": [float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])]
        })

    # create annotated image
    annotated = img_bgr.copy()
    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"{d['label']} {d['confidence']:.2f}", (x1, max(15, y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # encode to JPEG and base64
    _, img_encoded = cv2.imencode('.jpg', annotated)
    annotated_b64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')

    return detections, annotated_b64

# -------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "model_path": MODEL_PATH}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    confidence: float = Form(CONFIDENCE_DEFAULT)
):
    content = await file.read()
    img = read_imagefile(content)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    detections, annotated_b64 = run_inference_and_annotate(img, conf=confidence)

    return {
        "detections": detections,
        "annotated_image": annotated_b64,
        "timestamp": datetime.utcnow().isoformat()
    }

# -------------------------------------------------------------------------
# optional: simple history or debug endpoints can be added later
