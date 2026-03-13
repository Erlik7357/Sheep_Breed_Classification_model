<div align="center">

# SheepBreedNet

### Automated sheep breed identification using transfer learning with MobileNetV2.

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-2.x-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-API-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)

![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=plotly&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Training Pipeline](#training-pipeline)
- [Results and Evaluation](#results-and-evaluation)

---

## Overview

This project implements a **multi-class image classification model** to identify sheep breeds from photographs. Using the power of **transfer learning**, the MobileNetV2 model — pre-trained on ImageNet — is fine-tuned on a custom sheep dataset to deliver high accuracy with minimal training time and computational cost.

| Feature | Details |
|---|---|
| Problem Type | Multi-Class Image Classification |
| Dataset Source | Kaggle |
| Base Model | MobileNetV2 (ImageNet Weights) |
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |
| Training Strategy | Transfer Learning + Fine-Tuning |

---

## Dataset

The dataset was downloaded programmatically using the **Kaggle API**.

- **Source:** Kaggle Dataset Repository
- **Access Method:** Kaggle API (`kaggle.json` credentials)
- **Format:** Image files organized in class-named directories
- **Split:** Training / Validation / Test sets

> To use the Kaggle API, place your `kaggle.json` file in `~/.kaggle/` (Linux/Mac) or `C:\Users\<username>\.kaggle\` (Windows).

---

## Model Architecture

<div align="center">

```
Input Image (224 x 224 x 3)
        |
        v
+------------------------+
|   MobileNetV2 Base     |  <-- Pre-trained on ImageNet (Layers Frozen)
+------------------------+
        |
        v
+------------------------+
|  Global Average Pooling|
+------------------------+
        |
        v
+------------------------+
|    Dense Layer         |  <-- Custom Classification Head
|  (Softmax Activation)  |
+------------------------+
        |
        v
  Sheep Breed Prediction
```

</div>

**MobileNetV2** uses an inverted residual structure with linear bottlenecks, providing a great balance between accuracy and efficiency. It is ideal for deployment in resource-constrained environments.

---

## Tech Stack

<div align="center">

| Library / Tool | Purpose |
|---|---|
| ![TF](https://img.shields.io/badge/-TensorFlow-FF6F00?logo=tensorflow&logoColor=white) | Deep learning framework for building and training the model |
| ![Keras](https://img.shields.io/badge/-Keras-D00000?logo=keras&logoColor=white) | High-level API for model definition, compilation, and callbacks |
| ![Kaggle](https://img.shields.io/badge/-Kaggle_API-20BEFF?logo=kaggle&logoColor=white) | Programmatic dataset download |
| ![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white) | Array manipulation and numerical operations |
| ![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white) | Data loading and preprocessing |
| ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?logo=plotly&logoColor=white) | Plotting training curves and prediction results |
| ![Scikit](https://img.shields.io/badge/-Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white) | Train/test split and evaluation metrics |

</div>

---

## Project Structure

```
SheepBreedNet/
|
|-- Sheep_Breed_Classification.ipynb   # Main notebook (data loading, training, evaluation)
|-- labels.csv                         # Class label mappings for all sheep breeds
|-- sheep_breed_predictions.csv        # Model prediction outputs on the test set
|-- LICENSE                            # Project license
`-- README.md                          # Project documentation
```

---

## Setup and Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-username/SheepBreedNet.git
cd SheepBreedNet
```

**2. Install dependencies**

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn kaggle
```

**3. Configure Kaggle API**

Place your `kaggle.json` file in the correct directory:

```bash
# Windows
C:\Users\<username>\.kaggle\kaggle.json

# Linux / Mac
~/.kaggle/kaggle.json
```

**4. Download the dataset**

```bash
kaggle datasets download -d <dataset-name> --unzip -p ./data
```

---

## Training Pipeline

```text
Step 1  -->  Load and preprocess images (resize to 224x224, normalize)
Step 2  -->  Create TensorFlow data batches (Training / Validation / Test)
Step 3  -->  Load MobileNetV2 without top layer (ImageNet weights)
Step 4  -->  Freeze base model weights
Step 5  -->  Add Global Average Pooling + Dense (Softmax) head
Step 6  -->  Compile with Adam optimizer and Categorical Crossentropy loss
Step 7  -->  Train with EarlyStopping (monitor: val_loss, patience: 3)
Step 8  -->  Log metrics to TensorBoard
Step 9  -->  Evaluate on test set and visualize predictions
```

---

## Results and Evaluation

After training, the model is evaluated on the held-out test set. Key metrics tracked:

- **Training Accuracy**
- **Validation Accuracy**
- **Test Accuracy**
- **Loss Curve** (Training vs. Validation)

TensorBoard is used to visualize all metrics during the training process.

```bash
# Launch TensorBoard
tensorboard --logdir ./logs
```

---

<div align="center">

Made with dedication for deep learning and computer vision research.

</div>

