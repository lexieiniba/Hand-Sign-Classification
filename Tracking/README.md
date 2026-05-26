# Hand Sign Classification System

A real-time computer vision pipeline that detects hand gestures and classifies them using a custom-trained neural network.

## Project Overview
This system processes live webcam feeds to detect a hand, normalizes the input, and classifies it against a pre-trained Keras model. It is designed to be a scalable, modular pipeline for gesture-based interfaces.



## Technical Highlights
* **Landmark Extraction:** Utilizes MediaPipe to map 21-point spatial coordinates, treating the hand as a skeletal structure rather than a flat image.
* **Feature Engineering:** Transforms raw video into structured numerical formats by calculating spatial distances, joint angles, and temporal vectors, ensuring the model remains accurate even when the hand moves relative to the camera.
* **Computer Vision Pipeline:** Implemented a low-latency real-time tracking pipeline using `OpenCV` and `cvzone` to extract, normalize, and classify gestures at high frame rates.



## Architecture
1. **Detection & Tracking:** The system isolates the Region of Interest (ROI) using the Hand Tracking module.
2. **Normalization:** Detected hand images are mapped onto a standardized white canvas (`imgWhite`), ensuring consistent feature scaling.
3. **Inference:** A TensorFlow/Keras model performs real-time classification, mapping the geometric input to specific labels.
