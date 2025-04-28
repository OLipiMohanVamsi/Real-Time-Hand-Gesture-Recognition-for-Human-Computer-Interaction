# Real-Time Hand Gesture Recognition for Human-Computer Interaction

This project implements a real-time hand gesture recognition system for human-computer interaction (HCI) using deep learning techniques. The system enables seamless and intuitive control of digital devices through hand gestures, eliminating the need for physical touch. By leveraging Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks, the model can classify dynamic hand gestures in real-time, making it suitable for applications such as virtual reality (VR), smart homes, and assistive technologies.

## Problem Statement

With the increasing reliance on touchless interfaces in fields such as virtual reality, gaming, smart homes, and assistive devices, there is a growing need for systems that can recognize human gestures in real-time. Traditional HCI methods often rely on keyboards, mice, or touch screens, which limit accessibility and user interaction, especially for people with disabilities.

This project addresses the problem of building an accurate, real-time gesture recognition system that can be used in various applications, including:
- **Virtual Reality (VR):** Gesture-based control in immersive environments.
- **Smart Homes:** Touchless control of smart devices such as lights, thermostats, and speakers.
- **Assistive Technologies:** Providing people with disabilities a hands-free way to interact with digital devices.

## Dataset

For training and evaluation, the project utilizes the **Jester Gesture Dataset** from Kaggle, which contains **148,092 labeled video clips** of hand gestures, classified into **27 different gesture categories**. Each video clip consists of a sequence of RGB frames captured at 30 frames per second, with each gesture label associated with one of the 27 classes (e.g., "Swiping Left", "Zooming In", etc.).

- **Dataset URL:** [Jester Dataset on Kaggle](https://www.kaggle.com/datasets/20bn/jester)

## Methodology

The model combines two key deep learning components to achieve high performance:

1. **Convolutional Neural Networks (CNN):**
   - CNNs are used to extract spatial features from individual frames of the gesture videos. In particular, the **MobileNetV2** architecture is employed due to its lightweight and efficient nature, suitable for real-time applications.
   - The CNN extracts essential spatial features, such as hand shapes and movements, while reducing computational costs through depth-wise separable convolutions.

2. **Long Short-Term Memory (LSTM):**
   - The LSTM network is used to model the temporal dependencies between frames in a video sequence. Since hand gestures are dynamic and evolve over time, LSTM layers help capture the sequence information, ensuring that the model can recognize gestures accurately across multiple frames.
   
The hybrid CNN-LSTM architecture allows the model to process both spatial and temporal aspects of hand gestures.

## Results

- **Test Accuracy:** The model achieves an accuracy of over **90%** on the test set.
- **Inference Speed:** The model processes frames at **less than 33ms per frame**, meeting the real-time requirements (30 FPS).
- **Precision and Recall:** High performance across all gesture classes, with balanced precision, recall, and F1-scores.

The system can classify gestures in real-time, making it suitable for interactive applications in virtual environments and smart homes.

## Future Work

While this project demonstrates a strong foundation for real-time gesture recognition, several improvements can be made:
- **Multi-Hand Gesture Recognition:** Extend the model to recognize gestures involving multiple hands.
- **Mobile Optimization:** Further optimize the model for deployment on mobile and edge devices with limited resources.
- **Environmental Robustness:** Improve the model's performance under different lighting conditions and backgrounds through data augmentation and model enhancement.

## Project Structure

The project is organized into the following key files:

- **`train_model.py`**: Contains the code for building, training, and evaluating the CNN-LSTM model.
- **`real_time_recognition.py`**: Implements real-time gesture recognition using the webcam and displays predicted gestures.
- **`requirements.txt`**: Lists the necessary Python dependencies to run the project.
- **`README.md`**: Documentation describing the project and setup instructions.

## Setup and Installation

### Prerequisites

Ensure that the following dependencies are installed:

- Python 3.6+
- TensorFlow 2.x
- OpenCV
- MediaPipe
- NumPy
- Keras
- scikit-learn
- Matplotlib

