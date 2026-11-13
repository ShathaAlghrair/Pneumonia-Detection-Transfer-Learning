# Pneumonia Detection from Chest X-ray Images Using Transfer Learning

## Project Overview

This project presents a Deep Learning approach for automated pneumonia detection from chest X-ray images using Transfer Learning.

The proposed model is based on the pre-trained **VGG16** architecture, adapted for binary image classification to distinguish between normal lungs and pneumonia-affected lungs.

The study focuses on improving classification performance through image preprocessing, data augmentation, and class weighting techniques.

## Dataset

The project uses the publicly available **Chest X-Ray Images (Pneumonia)** dataset from Kaggle.

- **Dataset:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Total Images:** 5,863 chest X-ray images.
- **Classes:** Normal and Pneumonia (bacterial/viral).
- **Training Set:** 5,216 images.
- **Test Set:** 624 images.
- **Validation Set:** 10% of the training set.

## Methodology

### 1. Data Preprocessing
- Image resizing to 224 × 224 pixels.
- Pixel normalization to the range [0, 1].
- Preparation of training, validation, and test datasets.

### 2. Data Augmentation

Keras ImageDataGenerator was used to improve model generalization and reduce overfitting through:
- Random rotation (±15°).
- Random zoom (10%).
- Horizontal flipping.
- Shear transformation (10%).
- Nearest-neighbor filling.

### 3. Model Architecture – VGG16

The project uses VGG16 pre-trained on ImageNet as a feature extractor.

The custom classification head consists of:
- GlobalAveragePooling2D.
- Dense layer (256 units, ReLU activation).
- Dropout layer (0.5).
- Final Dense layer with Sigmoid activation for binary classification.

Class weighting was also applied to address dataset imbalance.

### 4. Training Configuration

- **Optimizer:** Adam.
- **Learning Rate:** 0.0001.
- **Loss Function:** Binary Cross-Entropy.
- **Batch Size:** 32.
- **Callbacks:** EarlyStopping and ModelCheckpoint.
- **Training:** Up to early stopping, with approximately 15 epochs reported.

## Results

The proposed VGG16-based model achieved the following results on the independent test set:

| Evaluation Metric | Performance |
|---|---:|
| Accuracy | 94.5% |
| Precision | 93.8% |
| Recall | 96.2% |
| F1-Score | 95.0% |
| ROC-AUC | 0.97 |

The model correctly classified 598 out of 624 test images, with 26 misclassifications.

## Technologies & Tools

- Python
- Deep Learning
- TensorFlow / Keras
- VGG16
- Transfer Learning
- Convolutional Neural Networks (CNN)
- Image Processing
- Scikit-learn
- Computer Vision

## Authors

- **Shatha Tayseer Alghrair**
- **Toleen Read Hijjawi**

Both authors contributed equally to this research.

## Limitations

- The study uses a single public dataset.
- The dataset does not distinguish between bacterial and viral pneumonia.
- External validation using multi-source clinical data is needed.
- Model explainability was not explored.

## Disclaimer

This project is an academic research study intended for educational purposes. It is not a substitute for professional medical diagnosis or clinical decision-making.
