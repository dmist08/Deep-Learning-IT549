# IT549: Deep Learning - Lab 3
## Image-Based AQI Classification using CNN and Pretrained Models

**Name:** Dharmikkumar Kamleshbhai Mistry <br> 
**Student ID:** 202511039

---

## Project Overview

End-to-end image classification pipeline to predict **AQI class** from location images, comparing a Custom CNN trained from scratch against MobileNetV2 fine-tuned via transfer learning.

---

## Dataset

- **Source:** [Google Drive](https://drive.google.com/drive/folders/1usBxgNB67GfhCQ2f7xRkDlF6fgIZZrP?usp=sharing)
- **Files:** `data.csv` (6000 rows) + `sampled_images/` folder
- **Columns used:** `Filename` (image path), `AQI_Class` (target - 6 classes)
- **Split:** 70% Train / 15% Validation / 15% Test (stratified)

---

## Results

| Metric | Basic CNN (Scratch) | MobileNetV2 (Transfer Learning) |
|---|---|---|
| Accuracy | 0.7344 | **0.9756** |
| Precision (macro) | 0.7418 | **0.9758** |
| Recall (macro) | 0.7344 | **0.9756** |
| F1-Score (macro) | 0.7258 | **0.9756** |

MobileNetV2 outperforms the Basic CNN by ~24% across all metrics.

---

## Tasks

| Task | Description |
|---|---|
| Task 1 | Data preparation - loading, encoding, stratified split, transforms |
| Task 2 | Custom CNN (3 conv blocks, ~57K parameters) trained from scratch |
| Task 3 | MobileNetV2 - Phase 1 frozen backbone, Phase 2 top-layer fine-tuning |
| Task 4 | Evaluation - Accuracy, Precision, Recall, F1, Confusion Matrix |
| Task 5 | Training curves + analysis of CNN vs MobileNetV2 behaviour |
| Task 6 | Misclassification analysis - 10 images with root cause explanation |

---

## Repository Structure

```
Lab03/
├── 202511039_lab03.ipynb
├── model_comparison.csv
└── README.md
```

> `sampled_images/` and `.pth` model weights are not pushed due to size.

---

## How to Run

```bash
# Local
pip install torch torchvision scikit-learn matplotlib seaborn pandas pillow
jupyter notebook IT549_Lab3_AQI_Classification.ipynb
```

---

## Key Findings

- MobileNetV2 reaches **97.56% accuracy** vs CNN's **73.44%** - a 24% gap driven by pretrained ImageNet features
- `c_Unhealthy_for_Sensitive_Groups` was the hardest class: CNN F1 = 0.47, MobileNetV2 F1 = 0.96
- All misclassifications occur between **adjacent AQI classes** - never between distant ones
- Several misclassified images are visually indistinguishable even to a human observer, suggesting label-image ambiguity in the dataset

---
