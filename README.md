# Smart-Traffic-Light-and-lane-Control-System-Using-Computer-Vision-for-intersections


Overview

Efficient traffic management at intersections is a cornerstone of urban mobility, safety, and sustainability. This project presents a Smart Traffic Light and Lane Control System leveraging Computer Vision (CV) to analyze real-time traffic conditions, detect vehicles, pedestrians, and traffic lights, and optimize traffic flow dynamically.

The system is designed for intersections and uses state-of-the-art object detection models to provide actionable traffic control decisions, reducing congestion and enhancing road safety. This project demonstrates the potential of deep learning and computer vision in intelligent transportation systems (ITS).

Features

Real-time Object Detection: Detects vehicles (cars, trucks, bikes), pedestrians, and traffic lights with high accuracy.

Traffic Light Classification: Differentiates traffic light states (red, green, yellow) including directional lights.

Lane Analysis: Monitors lane occupancy and traffic density to support intelligent signal timing.

Inference Web App: Streamlit-based UI allows uploading intersection images to visualize model predictions.

Scalable Architecture: Modular Python code enables easy integration with larger traffic management systems.

Dataset and Model

Original Dataset: Udacity traffic dataset (11 classes).

Class Remapping: Reduced to 6 target classes:

car (car, truck, biker)

person (pedestrian)

traffic_light

traffic_light_green

traffic_light_red

traffic_light_yellow

Model Architecture: YOLO11X (Ultralytics implementation) fine-tuned on remapped classes.

Trained Model Size: 335 MB (tracked via Git LFS).

Installation

Prerequisites:

Python 3.10+

Virtual environment (recommended)

# Cloning the repository
git clone https://github.com/N7Talha/Smart-Traffic-Light-and-lane-Control-System-Using-Computer-Vision-for-intersections.git
cd Smart-Traffic-Light-and-lane-Control-System-Using-Computer-Vision-for-intersections

# Create and activate virtual environment
python -m venv CVvenv
.\CVvenv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Ensure Git LFS is installed for the model
git lfs install


Usage
1. Training the Model
python train.py --dataset "D:\Work\Atomcamp\Final Project\project_dataset" --epochs 100 --imgsz 640 --batch 8


--dataset: Path to dataset

--epochs: Number of training epochs

--imgsz: Input image size (640x640 recommended)

--batch: Batch size (4–8 recommended depending on GPU)

2. Running Inference via Streamlit UI
streamlit run app.py


Upload an intersection image.

View real-time detection results including vehicles, pedestrians, and traffic lights.

3. Using the Model in Scripts
from ultralytics import YOLO

model = YOLO("model/best.pt")
results = model.predict("test_image.jpg")
results.show()

Repository Structure
├── train.py                 # Training script
├── traffic_logic.py         # Traffic control decision logic
├── traffic_system_inference # Scripts for inference
├── traffic_light_config.yaml
├── traffic_light_fixed.yaml
├── model/
│   └── best.pt              # Trained YOLO11X model
├── requirements.txt
├── app.py                   # Streamlit web app
└── README.md

Key Contributions

Developed a pipeline for class remapping, dataset preprocessing, and YOLO model training.

Designed an interactive Streamlit interface for non-technical users.

Demonstrated practical computer vision applications in traffic management.

Future Work

Integrate real-time video feed for continuous intersection monitoring.

Expand lane detection capabilities for multi-lane intersections.

Implement adaptive signal timing algorithms based on traffic flow predictions.

Explore edge deployment for real-time embedded systems in smart cities.

References

Ultralytics YOLO: https://github.com/ultralytics/ultralytics

Udacity Self-Driving Car Dataset: https://github.com/udacity/self-driving-car

Redmon, J., et al. "You Only Look Once: Unified, Real-Time Object Detection." CVPR, 2016.
