# FalconEye — Adaptive Object Detection for Space Station Safety

## 🚀 About FalconEye

FalconEye is a real-time object detection application built to safeguard astronauts in the zero-gravity environment of space stations. Equipped with a YOLOv8 model trained on a mission-specific dataset, it can accurately detect critical items — like **oxygen cylinders, toolboxes, and fire extinguishers** — even in dynamic, unpredictable conditions.  
## ⭐Features   

- Real-time detection via webcam feed or static images  
- Audio announcements of detected objects with confidence scores  
- Model selection dropdown for easy updates  
- Display of total objects detected per frame and saves results  
- User-friendly UI designed for non-technical users in critical environments  
- Accurate detection in dynamic, zero-gravity conditions
## App Demo  
![FalconEye Demo](docs/FalconEye_Demo.gif)



## 🛠 Dataset & Training  
- Dataset: Falcon simulated synthetic dataset targeting oxygen cylinders, toolboxes, and fire extinguishers  
- YOLOv8 model trained with optimized hyperparameters: 100 epochs, mosaic scheduling, early stopping  
- Performance:  
  - mAP@50: 94.4%  
  - mAP@50–95: 86.5%  
  - Precision: 97.3%  

## 🔍 Model Optimization  
- Data augmentation to simulate real-world conditions:  
  - HSV shifts (Hue 0.015, Saturation 0.7, Value 0.4) for lighting variations
  - Rotation ±10°, Scaling 0.5, Translation 10%, Shear 2  

## 📊 Testing & Evaluation  
- Tested on 400 simulated images  
- Metrics:  
  - mAP@50: 84.7%  
  - mAP@50–95: 73.4%  
  - Precision: 91.3%  

## 🖥 Application Development  
- Built with Streamlit for fast prototyping and a clean UI  
- Supports live webcam and image upload detection through browse files and example images  
- Lightweight and suitable for deployment in constrained environments  

## ⚙️ How to Run  
1. Clone this repository  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt

## Launch the app with Streamlit:
  streamlit run app.py  
- Use the webcam, upload your own images, or select from built-in example images to detect mission-critical objects in real-time

## 🎯 Future Enhancements  

- **Multi-Class Expansion:** Extend detection beyond oxygen tanks, fire extinguishers, and toolboxes to include astronaut suits, panels, leaks, waste items, and unauthorized objects — creating a complete space inventory and anomaly monitoring system.  

- **Dynamic Scenario Training:** Use Falcon Editor to simulate challenging space conditions like lighting shifts, occlusion, moving tools, and emergency drills. Continuously retrain the model on evolving mission scenarios for improved generalization.  

- **Cloud & Mission Control Integration:** Deploy the system on cloud servers for real-time monitoring from ground stations. Enable mission control to review logs, receive alerts, and analyze safety compliance remotely.  

 
# 🧑‍💻 Contributors
Team: **SnapCode**  
Team Leader: **Ipshita Sethi**  
Team Members: **Saumya Sudha,Lavanya Sharma,Muskan Goel**  

   
   

