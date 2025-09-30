# Handwritten Digit Recognition Model ✍️

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation & Usage](#installation--usage)
- [Contributing](#contributing)
- [Author](#author)
- [Contact](#contact)

## Introduction
Recognizing handwritten digits is a foundational problem in machine learning. This project presents a multi-class classification model designed to accurately identify digits from 0 to 9 using the MNIST dataset. The model employs a Convolutional Neural Network (CNN), which is particularly effective for image-based tasks. The architecture is carefully constructed with convolutional and max-pooling layers for robust feature extraction, along with dropout layers to prevent overfitting, resulting in a highly accurate and generalizable classifier.
## Features  
- **CNN Architecture:** Implements a modern CNN architecture with multiple convolutional and max-pooling layers to automatically learn hierarchical features from pixel data.
- **Regularization:** Utilizes Dropout layers to prevent overfitting and improve the model's performance on unseen data.
- **Multi-Class Classification:**  Capable of classifying images into one of ten categories (digits 0-9) using a softmax activation function.
- **High Performance:** Achieves a test accuracy of over 99%, demonstrating its effectiveness and robustness.
- **Built with TensorFlow/Keras:** Developed using the popular and powerful TensorFlow library

## Installation & Usage
**1. Clone the repository:**
```bash
git clone https://github.com/webserver105/Handwritten-Digit-Recognition-
cd handwritten-digit-recognition
```
**2. Installation:**
```bash
python -m venv venv
.\venv\Scripts\activate
pip install numpy matplotlib tensorflow
```

**3. Usage:**
```bash
python main.py
```

## Contributing
This project is open source so others can easily get involved. If you'd like to contribute, please fork the repository, create a feature branch, and open a pull request. All kinds of contributions bug fixes, features, or suggestions — are welcome!

## Author
Kunal Gandvane, kunal7sr@gmail.com\
Student at Department of Civil Engineering\
Indian Institute of Technology Bombay

## Contact
For any inquiries or further information, please contact me at [LinkedIn.](https://www.linkedin.com/in/kunal-gandvane-b28062346/)
