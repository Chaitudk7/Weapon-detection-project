AI-Powered Threat Detection Using Surveillance Camera (Weapon Detection)
📌 Overview

This project implements an AI-powered real-time weapon detection system designed to enhance public safety and automated surveillance. Using deep learning and computer vision, the system detects firearms, knives, and rifles from camera surveillance footage and triggers alerts for immediate response.

The goal is to support crime prevention, rapid threat awareness, and assist law enforcement through automated monitoring.

🧠 Key Features

Real-time Weapon Detection using YOLOv5

Detects guns, knives, and rifles with bounding boxes and confidence scores

90%+ accuracy on test data

Instant Audio Alert System for confirmed threats

User-friendly Gradio Web Interface for uploading images and receiving results

Scalable for live CCTV surveillance integration

🚀 Technologies Used
Category	Tools/Frameworks
Model	YOLOv5
Language	Python
Libraries	OpenCV, PyTorch, NumPy, Gradio
Deployment Interface	Gradio Web UI
Alerts	Python Audio Alert System
⚙️ How It Works

Input is taken from:

Uploaded image

Or live surveillance feed (optional future extension)

YOLOv5 processes the frame and detects weapons.

Bounding boxes and confidence scores highlight detected threats.

Audio alert triggers if the confidence threshold is met.

Output is displayed via a clean Gradio UI.

📁 Project Structure
├── yolov5/                     # Model and training scripts
├── models/                     # Trained weights
├── app.py                      # Main detection application
├── requirements.txt            # Dependencies
├── interface/                  # Gradio UI components
└── README.md                   # Documentation

🧪 Results

Achieved >90% accuracy

Detects weapons in varied lighting, angles, and real surveillance conditions

Fast inference suitable for real-time CCTV applications

🎯 Future Improvements

Live multi-camera feed support

SMS/Email alert system

Edge deployment (Jetson Nano / Raspberry Pi + Coral TPU)

Expanded dataset for more weapon classes

📌 Use Cases

Smart public surveillance

Restricted zone monitoring

School and campus security

Law enforcement automation

Military facilities monitoring
