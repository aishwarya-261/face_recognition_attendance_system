---
title: Face Recognition Attendance System
emoji: 🎓
colorFrom: indigo
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
---

# 🎓 Face Recognition Attendance System

A robust, deep-learning-based attendance system using **MTCNN** for face detection and **FaceNet (InceptionResnetV1)** for face recognition. This project features both a local Tkinter GUI and a web-based Gradio interface for easy deployment on Hugging Face Spaces.

## 🚀 Features

- **Real-time Face Detection**: Uses MTCNN to accurately locate faces.
- **High-Accuracy Recognition**: Powered by FaceNet pre-trained on VGGFace2.
- **Data Augmentation**: Automatically generates augmented images for better training.
- **Dual Interface**:
  - `main.py`: Local desktop application (Tkinter).
  - `app.py`: Web-based application (Gradio) – ideal for deployment.
- **CSV Logging**: Automatically logs attendance with timestamps.

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/aishwarya-261/face_recognition_attendance_system.git
   cd face_recognition_attendance_system
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **System Dependencies** (Linux/Hugging Face):
   Ensure you have the following installed (handled by `packages.txt` on HF):
   ```bash
   libgl1-mesa-glx
   libglib2.0-0
   ```

## 💻 Usage

### Local Desktop App
Run the Tkinter GUI:
```bash
python main.py
```

### Web App / Hugging Face
Run the Gradio interface:
```bash
python app.py
```

## 🌐 Deployment on Hugging Face Spaces

1. Create a new **Space** on [Hugging Face](https://huggingface.co/spaces).
2. Select **Gradio** as the SDK.
3. Upload all files from this repository (including `app.py`, `requirements.txt`, and `packages.txt`).
4. The space will automatically install dependencies and start the app.

## 📊 Evaluation
You can evaluate the model accuracy and plot comparisons using:
- `evaluate_accuracy.py`
- `plot_comparison.py`

## 📄 License
MIT License
