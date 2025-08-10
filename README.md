# lconEye — Adaptive Object Detection for Space Station Safety

## 🚀 Problem Statement  
In the zero-gravity environment of a space station, vital items like oxygen cylinders, fire extinguishers, and toolboxes can drift or become misplaced, posing serious safety risks. Current monitoring methods lack the intelligence and immediacy needed for real-time prevention.

## 💡 Solution Overview  
FalconEye is an adaptive object detection system using a YOLOv8 model trained on a mission-specific synthetic dataset. It provides:

- Real-time detection via webcam feed or static images  
- Audio announcements of detected objects with confidence scores  
- Model selection dropdown for easy updates  
- Display of total objects detected per frame and saved results  
- User-friendly UI designed for non-technical users in critical environments  
- Accurate detection in dynamic, zero-gravity conditions  

## 🛠 Dataset & Training  
- Dataset: Falcon simulated synthetic dataset targeting oxygen cylinders, toolboxes, and fire extinguishers  
- YOLOv8 model trained with optimized hyperparameters: 100 epochs, mosaic scheduling, early stopping  
- Performance:  
  - mAP@50: 94.4%  
  - mAP@50–75: 86.5%  
  - Precision: 97.3%  

## 🔍 Model Optimization  
- Data augmentation to simulate real-world conditions:  
  - HSV shifts (Hue 0.015, Saturation 0.7, Value 0.4)  
  - Rotation ±10°, Scaling 0.5, Translation 10%, Shear 2  

## 📊 Testing & Evaluation  
- Tested on 400 simulated images  
- Metrics:  
  - mAP@50: 84.7%  
  - mAP@50–95: 73.4%  
  - Precision: 91.3%  
- Retesting improved results to mAP@50: 93.8%, mAP@50–95: 79.5%  

## 🖥 Application Development  
- Built with Streamlit for fast prototyping and a clean UI  
- Supports live webcam and image upload detection  
- Lightweight and suitable for deployment in constrained environments  

## ⚙️ How to Run  
1. Clone this repository  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt

# Launch the app with Streamlit:
streamlit run app.py

Use the webcam, upload your own images, or select from built-in example images to detect mission-critical objects in real-time

# Future Enhancements
Multi-model integration: Incorporate additional YOLO versions or alternative detection models to broaden the range of detectable objects, enhancing adaptability to new mission requirements.

Advanced audio alerts: Develop a more sophisticated notification system with customizable urgency levels and spatial audio cues to better support real-time decision-making in critical situations.

Embedded system optimization: Refine the application for deployment on low-power, embedded hardware typical in space stations, ensuring efficient performance without sacrificing detection accuracy.

# 🧑‍💻 Contributors
[Your Name / Team Name]

   
   

