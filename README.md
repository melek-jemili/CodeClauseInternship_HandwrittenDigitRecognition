# CodeClauseInternship_HandwrittenDigitRecognition
A system that can recognize sequences of handwritten digits
# 🎯 Handwritten Digit Recognition System

A comprehensive deep learning solution for recognizing handwritten digits (0-9) using Convolutional Neural Networks (CNN). This system can handle both single digits and sequences of digits with high accuracy and provides multiple user-friendly interfaces for testing.

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Features

### Core Functionality
- **High Accuracy**: Achieves >98% accuracy on MNIST dataset
- **Single Digit Recognition**: Predict individual handwritten digits
- **Sequence Recognition**: Automatically segment and recognize multiple digits
- **Real-time Processing**: Fast prediction with confidence scores
- **Model Persistence**: Save and load trained models

### User Interface Options
- 🖼️ **Image File Testing**: Load and test any image file
- 📷 **Webcam Capture**: Real-time recognition from camera
- 🎨 **Interactive Drawing**: Draw digits with mouse/touchpad
- 📝 **Sequence Processing**: Handle images with multiple digits
- 📁 **Batch Testing**: Process entire folders of images
- 🧪 **MNIST Validation**: Test against standard dataset

## 🏗️ Architecture

### Model Structure
```
Input (28×28×1) → Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(64) → Flatten → Dense(64) → Dropout → Dense(10)
```

### Key Components
- **Convolutional Layers**: Feature extraction from digit images
- **Max Pooling**: Dimensionality reduction and translation invariance
- **Dense Layers**: Classification with dropout for regularization
- **Softmax Output**: Probability distribution over 10 digit classes

## 🚀 Quick Start

### Prerequisites
```bash
pip install tensorflow opencv-python matplotlib pillow numpy
```

### Installation
1. Clone or download the project files
2. Install dependencies
3. Run the main script

```bash
python digit_recognition.py
```

### First Run
The system will automatically:
1. Download and preprocess MNIST data
2. Build the CNN model
3. Train for 10 epochs (~5-10 minutes)
4. Save the trained model
5. Launch the interactive menu

## 📖 Usage Guide

### Interactive Menu
After running the script, you'll see:
```
🎯 HANDWRITTEN DIGIT RECOGNITION - TEST INTERFACE
============================================================
1. 🖼️  Test single image file
2. 📷  Capture from webcam
3. 🎨  Draw digit with mouse
4. 📝  Test sequence of digits
5. 📁  Batch test multiple images
6. 🧪  Test with MNIST samples
7. 📊  Show model performance
8. 💾  Save/Load model
9. ❌  Exit
```

### Testing Methods

#### 1. Single Image Testing
```python
# Example usage in code
recognizer = HandwrittenDigitRecognizer()
digit, confidence = recognizer.predict_digit('path/to/image.png')
print(f"Predicted: {digit} (Confidence: {confidence:.3f})")
```

#### 2. Webcam Recognition
- Shows live camera feed
- Press SPACE to capture and predict
- Press ESC to exit
- Works with handwritten digits on paper

#### 3. Drawing Interface
- Interactive matplotlib window
- Draw with mouse/trackpad
- Automatic prediction on window close
- Shows probability distribution

#### 4. Sequence Recognition
```python
# Process image with multiple digits
predictions = recognizer.predict_sequence('sequence.png')
for i, (digit, conf) in enumerate(predictions):
    print(f"Position {i}: {digit} ({conf:.3f})")
```

## 📊 Performance

### Training Results
- **Training Accuracy**: ~99.5%
- **Validation Accuracy**: ~98.8%
- **Training Time**: 5-10 minutes (CPU)
- **Inference Speed**: ~10ms per digit

### Model Metrics
- **Parameters**: ~93K trainable parameters
- **Model Size**: ~380KB saved file
- **Input**: 28×28 grayscale images
- **Output**: 10-class probability distribution

## 🛠️ Code Structure

### Main Classes

#### `HandwrittenDigitRecognizer`
Core recognition functionality:
```python
- load_and_preprocess_data()    # Load MNIST dataset
- build_model()                 # Create CNN architecture  
- train_model()                 # Train with callbacks
- predict_digit()               # Single digit prediction
- predict_sequence()            # Multiple digit recognition
- preprocess_image()            # Image preprocessing
- segment_digits()              # Digit segmentation
```

#### `DigitRecognitionUI`
Interactive user interface:
```python
- interactive_menu()            # Main menu system
- test_single_image()           # File testing interface
- capture_from_webcam()         # Camera interface
- draw_digit_interface()        # Drawing interface
- batch_test_images()           # Folder processing
- test_mnist_samples()          # MNIST validation
```

## 📁 File Structure

```
digit-recognition/
├── digit_recognition.py       # Main script
├── digit_recognition_model.h5  # Trained model (created after first run)
└── test_images/               # Sample test images (optional)
    ├── digit_0.png
    ├── digit_1.png
    └── sequence_123.png
```

## 🎯 Image Requirements

### Single Digits
- **Format**: PNG, JPG, JPEG, BMP, TIFF
- **Size**: Any (automatically resized to 28×28)
- **Background**: White background preferred
- **Digit Color**: Black or dark colors
- **Content**: Single digit, centered

### Digit Sequences
- **Layout**: Digits arranged horizontally
- **Spacing**: Clear separation between digits
- **Background**: Clean, uniform background
- **Quality**: Clear, non-overlapping digits

## 🔧 Customization

### Model Parameters
```python
# Modify in build_model()
Conv2D(filters=32, kernel_size=(3,3))  # Adjust filters
Dense(units=64)                        # Change dense layer size
Dropout(rate=0.5)                      # Adjust dropout rate
```

### Training Parameters
```python
# Modify in train_model()
epochs=10           # Training epochs
batch_size=128      # Batch size
patience=3          # Early stopping patience
```

### Preprocessing
```python
# Modify in preprocess_image()
target_size = (28, 28)    # Model input size
threshold_method = cv2.THRESH_OTSU  # Thresholding method
```

## 🐛 Troubleshooting

### Common Issues

#### Webcam Not Working
```bash
# Check camera permissions
# Try different camera indices
cap = cv2.VideoCapture(1)  # Try 1, 2, etc.
```

#### Low Accuracy on Custom Images
- Ensure white background, black digits
- Check image quality and contrast
- Verify digit is centered and clear
- Try different preprocessing parameters

#### Memory Issues
```python
# Reduce batch size
batch_size = 64  # Instead of 128

# Use model checkpointing for large datasets
callbacks = [tf.keras.callbacks.ModelCheckpoint('best_model.h5')]
```

#### Import Errors
```bash
# Update packages
pip install --upgrade tensorflow opencv-python matplotlib

# For Mac M1/M2
pip install tensorflow-macos tensorflow-metal
```

## 📈 Extending the Project

### Possible Enhancements
- **Multi-language Support**: Extend to other scripts (Arabic numerals, etc.)
- **Data Augmentation**: Add rotation, scaling, noise for robustness
- **Model Architectures**: Try ResNet, DenseNet, or Vision Transformers
- **Mobile Deployment**: Convert to TensorFlow Lite for mobile apps
- **Web Interface**: Create Flask/Django web application
- **Real-time Video**: Continuous digit recognition in video streams

### Advanced Features
```python
# Data augmentation example
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

# Model ensemble
models = [model1, model2, model3]
predictions = [model.predict(image) for model in models]
final_prediction = np.mean(predictions, axis=0)
```

## 📝 API Reference

### Core Methods

#### `predict_digit(image_path_or_array)`
Predict a single digit from image.
- **Input**: Image path (string) or numpy array
- **Returns**: Tuple (predicted_digit, confidence)
- **Example**: `digit, conf = recognizer.predict_digit('digit.png')`

#### `predict_sequence(image_path)`
Predict sequence of digits from image.
- **Input**: Path to image with multiple digits
- **Returns**: List of (digit, confidence) tuples
- **Example**: `results = recognizer.predict_sequence('123.png')`

#### `preprocess_image(image_path_or_array)`
Preprocess image for model input.
- **Input**: Image path or array
- **Returns**: Preprocessed array (1, 28, 28, 1)
- **Processing**: Resize, normalize, reshape

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Submit pull request with description

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to functions
- Include type hints where possible
- Write unit tests for new features

## 📄 License

This project is licensed under the MIT License - see below for details:

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

## 🙏 Acknowledgments

- **MNIST Dataset**: Yann LeCun, Corinna Cortes, Christopher J.C. Burges
- **TensorFlow Team**: For the excellent deep learning framework
- **OpenCV Community**: For computer vision tools
- **Contributors**: Thanks to all who helped improve this project

## 📞 Support

### Getting Help
- **Issues**: Open a GitHub issue with detailed description
- **Questions**: Check existing issues and documentation first
- **Feature Requests**: Create an issue with "enhancement" label

### Resources
- [TensorFlow Documentation](https://www.tensorflow.org/tutorials)
- [OpenCV Tutorials](https://docs.opencv.org/4.x/d9/df8/tutorial_root.html)
- [MNIST Dataset Info](http://yann.lecun.com/exdb/mnist/)

---

⭐ **Star this repository if you find it helpful!**

