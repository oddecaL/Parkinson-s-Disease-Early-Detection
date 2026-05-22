# Generalization of Audio-Based Early Parkinson’s Detection from Controlled Speech to Real-World Mobile Speech

## Overview

This project investigates early Parkinson’s Disease (PD) detection using voice recordings and neural networks.

The work extends the methodology presented in:

> Audio-Based Neural Network to Classify Patients With Early Parkinson’s Disease  
> IEEE Document ID: 11146786

Instead of focusing exclusively on controlled phonation tasks, this project evaluates whether speech-based PD detection can generalize to real-world mobile speech recordings.

Dataset used:

MDVR-KCL (Mobile Device Voice Recordings at King’s College London)

---

## Research Gap

Existing audio-based Parkinson’s detection approaches are primarily evaluated on controlled vocal tasks and provide limited validation under realistic mobile speech environments.

---

## Objective

To evaluate whether speech representations learned from structured vocal recordings remain effective for Parkinson’s Disease detection under real-world mobile speech conditions.

---

## Novelty

This work investigates:

- Controlled speech vs real-world speech performance
- Mobile-recorded voice robustness
- Spectrogram-based deep learning
- Acoustic feature comparison

---

## Dataset

Dataset:
MDVR-KCL (Zenodo Record: 2867216)

Classes:

- HC → Healthy Control
- PD → Parkinson’s Disease

Speech Tasks:

- ReadText
- SpontaneousDialogue

Recording Type:

- Mobile device recordings

Current Dataset Distribution:

| Task | HC | PD |
|------|----|----|
| ReadText | 21 | 16 |
| SpontaneousDialogue | 21 | 15 |

Total Audio Files: 73

---

## Environment

Google Colab

Python 3.12

Required packages:

```bash
pip install tensorflow
pip install keras
pip install librosa
pip install opencv-python
pip install seaborn
pip install zenodo-get
```

---

## Project Structure

```
26-29_09_2017_KCL/

├── ReadText/
│   ├── HC/
│   └── PD/

├── SpontaneousDialogue/
│   ├── HC/
│   └── PD/
```

---

## Methodology

### 1. Audio Loading

- Load `.wav` files
- Resample to 16 kHz

### 2. Spectrogram Generation

Generate:

- Mel Spectrogram
- Log Mel Spectrogram

Parameters:

- n_mels = 128
- fmax = 8000 Hz

### 3. Image Transformation

Convert spectrogram:

- Resize → 128 × 128
- Normalize → [0,1]

Output Shape:

```
(N,128,128,1)
```

---

## Deep Learning Pipeline

Model:

CNN

Architecture:

- Conv2D (32)
- MaxPooling
- Conv2D (64)
- MaxPooling
- Conv2D (128)
- MaxPooling
- Flatten
- Dense (128)
- Dropout
- Sigmoid Output

Training:

- Epochs = 30
- Batch Size = 16
- Optimizer = Adam

---

## Acoustic Feature Pipeline

Extracted Features:

- Duration
- MFCC Mean
- MFCC Std
- Spectral Centroid
- Spectral Rolloff
- Zero Crossing Rate
- RMS Energy
- Chroma Features

Total Features:

47

Classifier:

Random Forest

Parameters:

- n_estimators = 100
- max_depth = 15

---

## Current Results

### CNN

| Metric | Value |
|--------|-------|
| Accuracy | 0.750 |
| Precision | 1.000 |
| Recall | 0.333 |
| F1 Score | 0.500 |
| ROC AUC | 0.800 |

---

### Acoustic Feature Model

| Metric | Value |
|--------|-------|
| Accuracy | 0.875 |
| Precision | 1.000 |
| Recall | 0.667 |
| F1 Score | 0.800 |
| ROC AUC | 0.867 |

---

## Observations

- Acoustic feature engineering outperformed CNN on this small dataset.
- CNN showed signs of overfitting.
- Real-world speech conditions introduce variability.

---
