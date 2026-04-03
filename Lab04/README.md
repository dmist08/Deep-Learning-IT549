# IT549: Deep Learning - Lab 4

## Object Detection Evolution: From R-CNN to YOLO

**Name:** Dharmikkumar Mistry  
**Roll Number:** 202511039 

---

## Project Overview

This lab implements and compares the full evolution of object detection architectures:

| Task | Description |
|------|-------------|
| Preparation | Ground truth bounding box visualization from YOLO annotations |
| Task 1 | IoU (Intersection over Union) from scratch |
| Task 2 | Selective Search - R-CNN region proposals |
| Task 3 | R-CNN bottleneck - 100 crops × independent CNN passes (timed) |
| Task 4 | Fast R-CNN - single CNN pass + RoI Pooling (timed + compared) |
| Task 5 | Faster R-CNN - pre-trained model with RPN inference |
| Task 6 | Custom Non-Maximum Suppression (NMS) implementation |
| Task 7 | YOLOv8n fine-tuning on Fruit dataset + mAP evaluation |

---

## Dataset

**Fruit Images for Object Detection** (Kaggle: `mbkinaci/fruit-images-for-object-detection`)  
- 3 classes: Apple, Banana, Orange  
- 240 training images, 60 test images  
- Annotations in Pascal VOC XML format (converted to YOLO TXT format for training)

---

## Key Concepts Covered

- **IoU** - the fundamental overlap metric for detection evaluation
- **Selective Search** - external region proposal algorithm used in R-CNN
- **RoI Pooling** - Fast R-CNN's mechanism to share a single feature map across proposals
- **Region Proposal Network (RPN)** - Faster R-CNN's learned alternative to Selective Search
- **Non-Maximum Suppression (NMS)** - greedy algorithm to eliminate duplicate detections
- **YOLO** - single-stage regression detector, no proposals needed

---

## Files

```
Lab04/
├── lab04.ipynb          ← Main notebook (all tasks)
├── fruit_data.yaml      ← YOLO dataset config
├── yolov8n.pt           ← YOLOv8n base weights (downloaded)
└── README.md            ← This file
```

> **Note:** `data/`, `yolo_dataset/`, and `yolo_runs/` directories are excluded from version
> control (large files). Run the notebook cells sequentially to regenerate them.

---

## Environment

- Python 3.13
- PyTorch 2.6.0+cu124 / torchvision 0.21.0
- ultralytics 8.4.14 (YOLOv8)
- opencv-contrib-python (required for Selective Search)
- CUDA: NVIDIA GeForce RTX 2050 (4 GB VRAM)

Tested locally on Windows with CUDA. GPU is required for Task 7 YOLO training at practical speed.

---

## Results Summary

| Metric | Value |
|--------|-------|
| R-CNN (100 crops) | 0.4942s - 4.94 ms/crop |
| Fast R-CNN (1 pass + RoI Pool) | 0.0145s - **34.1× faster** |
| Faster R-CNN inference | 185.6 ms/image |
| YOLOv8n pre-trained inference | 22.0 ms/image |
| YOLOv8n fine-tuned inference | 15.8 ms/image |
| YOLOv8n fine-tuned mAP@50 | 0.8131 |
| YOLOv8n fine-tuned mAP@50-95 | 0.5912 |
| YOLOv8n fine-tuned Precision | 0.8331 |
| YOLOv8n fine-tuned Recall | 0.7012 |