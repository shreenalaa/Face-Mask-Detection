# 😷 Face Mask Detection System using Deep Learning (CNN)

A complete AI project for building a **Face Mask Detection System** using **Convolutional Neural Networks (CNN)** with **TensorFlow / Keras**.

---

## 📌 Project Overview

This project aims to:
- Classify images into two categories:
  - ✅ With Mask  
  - ❌ Without Mask
- Build a CNN model from scratch
- Train the model on real image data
- Evaluate model performance
- Implement a real prediction system for new images

---

## 🧠 Workflow Pipeline

1. Data loading  
2. Label creation  
3. Image visualization  
4. Image preprocessing  
5. Data conversion to NumPy arrays  
6. Dataset merging  
7. Train/Test split  
8. Data normalization  
9. CNN model building  
10. Model training  
11. Model evaluation  
12. Prediction system

---

## 📂 Dataset Structure

```text
data/
│
├── with_mask/
│   ├── with_mask_1.jpg
│   ├── with_mask_2.jpg
│   └── ...
│
└── without_mask/
    ├── without_mask_1.jpg
    ├── without_mask_2.jpg
    └── ...
🏷️ Label Encoding
Class	Label
With Mask	1
Without Mask	0
⚙️ Dependencies
Python 3.x
numpy
matplotlib
opencv-python
Pillow
scikit-learn
tensorflow
keras
Install:

pip install numpy matplotlib opencv-python pillow scikit-learn tensorflow keras
🖼️ Image Preprocessing
Resize images to 128 × 128

Convert to RGB

Normalize pixel values

Convert to NumPy arrays

Merge datasets

Encode labels

🧪 Data Splitting
80% Training
20% Testing
10% Validation (from training set)
🧠 CNN Architecture
Input: (128, 128, 3)

Conv2D(32) + ReLU
MaxPooling2D(2×2)

Conv2D(64) + ReLU
MaxPooling2D(2×2)

Flatten

Dense(128) + ReLU
Dropout(0.5)

Dense(64) + ReLU
Dropout(0.5)

Output:
Dense(2) + Sigmoid
⚡ Model Configuration
Optimizer: Adam
Loss: Sparse Categorical Crossentropy
Metric: Accuracy
📈 Training Setup
Epochs: 5
Validation Split: 0.1
📊 Evaluation
Training Loss

Validation Loss

Training Accuracy

Validation Accuracy

Accuracy vs Epoch plots

Loss vs Epoch plots

🔮 Prediction System
Features:
User inputs image path

Image displayed using OpenCV

Image resizing to model input size

Normalization

Model prediction

Class probability output

Final classification result

Output:
The person in the image is wearing a mask
OR
The person in the image is not wearing a mask
🚀 How To Run
python mask_detection.py
Then enter:

Path of the image to be predicted: path/to/image.jpg
🎯 Applications
Smart surveillance systems

Public safety monitoring

Healthcare AI systems

Smart cities

Computer vision applications

AI research projects

Academic projects

Real-world AI deployment

🔮 Future Enhancements
Transfer Learning (MobileNet, ResNet, EfficientNet)

Real-time detection (Webcam)

Face detection integration (MTCNN / Haar Cascade)

Edge AI deployment (Raspberry Pi / Jetson Nano)

Model optimization

Quantization

API deployment (FastAPI)

Web interface

Mobile app integration

👩‍💻 Author
Shereen Alaa
Machine Learning Engineer

GitHub: https://github.com/shreenalaa

LinkedIn: https://www.linkedin.com/in/shreen-alaa/

✨ A complete AI pipeline combining Computer Vision + Deep Learning + Model Training + Deployment Logic for real-world applications.
