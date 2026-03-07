# Bone Fracture Classification System

Deep learning-based web application for automated bone fracture detection from X-ray images using transfer learning.

## Project Overview

This project uses a pretrained CNN model to classify bone X-ray images into fracture and non-fracture categories. The system provides an easy-to-use web interface where users can upload X-ray images and receive real-time predictions.

The model is built using TensorFlow and deployed as a web application using Flask. The entire application is containerized with Docker and deployed on AWS for scalable inference.

## Model Architecture

The system uses transfer learning with the pretrained **VGG16** convolutional neural network trained on **ImageNet**.

Architecture:

VGG16 (Frozen Convolutional Layers)
→ Global Average Pooling
→ Dense Layer
→ Output Layer (Fracture / No Fracture)

Only the dense classification layers are trained while the convolutional layers remain frozen.

## Features

* Automated bone fracture detection
* Transfer learning using pretrained CNN
* Image preprocessing and augmentation
* Web interface for image upload and prediction
* REST API backend for inference
* Docker containerized deployment
* AWS cloud deployment

## Tech Stack

Python
TensorFlow
OpenCV
Flask
HTML / CSS
Docker
AWS

## Dataset

Medical X-ray image dataset containing fractured and non-fractured bone images. Data augmentation techniques such as rotation, flipping, and normalization were applied to improve model generalization.

## Model Performance

Accuracy: 92%
Evaluation Metrics: Accuracy, Precision, Recall

## Project Structure

web_app/

│
├── model/
│   └── fracture_model.h5
│
├── app.py
├── requirements.txt
├── Dockerfile
├── templates/
├── static/
└── README.md

## Installation

Clone the repository

git clone https://github.com/nithishkumar86/web_app.git

Install dependencies

pip install -r requirements.txt

Run the application

python app.py

## Docker Deployment

Build Docker image

docker build -t fracture-classifier .

Run container

docker run -p 5000:5000 fracture-classifier

## Future Improvements

* Add more medical datasets for improved accuracy
* Implement Grad-CAM visualization for explainable AI
* Deploy using scalable microservices architecture
