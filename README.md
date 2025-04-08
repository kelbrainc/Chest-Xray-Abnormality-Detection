# Chest-Xray-Abnormality-Detection

This kaggle competition develops an automated system for detecting abnormalities in chest x-ray images using deep learning. By leveraging bounding box annotations, custom data generators, and transfer learning with state-of-the-art CNN architectures, the system classifies images as "No Finding" (normal) or "Abnormal". An ensemble approach combining EfficientNet, ResNet, and DenseNet based models further enhances the overall performance.
Achieved highest test accuracy of 77.5%. link to kaggle competition https://www.kaggle.com/competitions/cs604xray/leaderboard?tab=private
---

## Table of Contents

- [Overview](#overview)
- [Dataset and Annotation](#dataset-and-annotation)
- [Pre-processing and Data Augmentation](#pre-processing-and-data-augmentation)
- [Model Architectures](#model-architectures)
  - [EfficientNet (Eff1)](#efficientnet-eff1)
  - [ResNet (ResNet1)](#resnet-resnet1)
  - [DenseNet (DenseNet1)](#densenet-densenet1)
- [Training and Inference](#training-and-inference)
- [Ensemble Strategy](#ensemble-strategy)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)


---

## Overview

This competition is designed to assist radiologists by automatically detecting abnormalities in chest x-ray images. It uses bounding box information to crop regions of interest from the raw images and then processes these images using deep neural networks pretrained on ImageNet. Three distinct architectures—EfficientNetB3, ResNet50, and DenseNet121—are employed. Finally, an ensemble combines their outputs using a weighted average to generate robust final predictions.

---

## Dataset and Annotation

- **Data Files:**
  - **Data_Entry_2017_v2020.csv:** Contains metadata for the x-ray images including file names and diagnostic labels.
  - **BBox_List_2017.csv:** Provides bounding box coordinates for images that require cropping to isolate abnormal regions.
- **Image Directories:**
  - **Training Images:** Located in a folder (e.g., `images/image_com`).
  - **Evaluation Images:** Stored in a separate folder (e.g., `eval_xray_im`).

A binary label is generated—images marked as `"No Finding"` are labeled 0, while all others are labeled 1 (abnormal).

---

## Pre-processing and Data Augmentation

- **Bounding Box Cropping:**  
  Custom data generators check for bounding box information. When available, an image is cropped to the provided coordinates.
  
- **Image Processing:**
  - All images are resized to (224, 224).
  - Grayscale images are converted to 3-channel RGB by repeating the single channel.
  - Architecture-specific preprocessing (using `preprocess_input` from the corresponding model library) is applied.

- **Data Augmentation (EfficientNet and DenseNet):**
  - Augmentations include rotations (up to 10°), width/height shifts (10%), shear (10°), and zoom (10%).
  - An instance of Keras’ `ImageDataGenerator` is used to generate variations during training for improved generalization.

- **Custom Data Generators:**  
  All models rely on generators inheriting from `Sequence` to efficiently load, process, and batch images for training and testing.

---

## Model Architectures

### EfficientNet

- **Base Model:**  
  EfficientNetB3 (pretrained on ImageNet) is used as the feature extractor.
  
- **Custom Head:**
  - A `GlobalAveragePooling2D` layer reduces the spatial dimensions.
  - Followed by batch normalization and dropout (0.3 rate) to mitigate overfitting.
  - A Dense layer with 1024 neurons (ReLU activation) is added.
  - A final softmax layer outputs probabilities for 2 classes (normal vs. abnormal).
  
- **Training:**  
  The model is trained using the Adam optimizer (learning rate = 1e-4) with callbacks such as `ReduceLROnPlateau` and `EarlyStopping`.

- **Output:**  
  After training, the model weights are saved, and predictions are generated on an evaluation set. The predictions are stored in a `.npy` file.

### ResNet

- **Base Model:**  
  ResNet50 (pretrained on ImageNet) is used as the backbone.
  
- **Custom Head:**
  - A `GlobalAveragePooling2D` layer is applied.
  - A Dense layer with 1024 neurons (ReLU activation) and a dropout (0.5 rate) is added.
  - The final layer is a softmax with 2 outputs.
  
- **Training:**  
  The model is compiled with SGD (learning rate = 0.001) and trained for 5 epochs using a custom data generator that also crops images by bounding box when available.
  

### DenseNet

- **Base Model:**  
  DenseNet121 (pretrained on ImageNet) is the chosen feature extractor.
  
- **Custom Head:**
  - Global average pooling followed by batch normalization and dropout (0.3) are applied.
  - A Dense layer with 1024 neurons (ReLU activation) and another dropout (0.3) precede the final softmax layer.
  
- **Training:**  
  The model is compiled using the Adam optimizer with a learning rate of 1e-4 and trained with similar augmentation strategies as EfficientNet.
  

---

## Training and Inference

- **Data Splitting:**  
  The dataset is split into training and validation sets (80/20) with stratification on the binary label.

- **Callbacks:**  
  Learning rate reduction and early stopping ensure efficient training.

- **Inference:**  
  A separate test data generator processes evaluation images. The trained models generate predictions, which are then saved to disk.

## Ensemble Strategy

- **Prediction Fusion:**  
  Final ensemble predictions are obtained by loading the saved predictions from the three models:
  - **Weights:**  
    - EfficientNet: 0.6  
    - DenseNet: 0.3  
    - ResNet: 0.1
  - **Weighted Average:**  
    Each model’s probability outputs are combined using the assigned weights.
  - **Thresholding:**  
    The ensemble output is thresholded (e.g., probability > 0.6 for abnormal) to assign the final binary label.
  
- **Submission:**  
  A submission DataFrame is prepared with image IDs and the final predicted labels.

---

## Installation & Setup

### Prerequisites

- **Python 3.7+**
- **TensorFlow (with Keras)**
- **Additional Libraries:** NumPy, Pandas, scikit-learn, Pillow (PIL), etc.


### Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd chest-xray-abnormality-detection
