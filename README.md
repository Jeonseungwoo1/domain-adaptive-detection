# Domain-Adaptive Object Detection with Faster R-CNN

This repository implements a domain-adaptive object detection system using Faster R-CNN, designed to study how object detectors perform when transferred from natural images to artistic domains (watercolor, clipart, comic).

## Overview

The project explores domain adaptation in object detection by training and evaluating Faster R-CNN models across different visual styles. It provides a clean, modular implementation for experimenting with various training strategies to understand how models generalize from photographic to artistic domains.

## Features

- **Multiple Training Modes**
  - **Zero-shot**: Direct evaluation of pre-trained COCO models on artistic domains
  - **Fine-tune**: Adapt pre-trained models to artistic domains
  - **Scratch**: Train from random initialization

- **Backbone Freezing Options**
  - Full freeze: Freeze entire ResNet backbone
  - Partial freeze: Only unfreeze layer4
  - BN only: Only unfreeze batch normalization layers
  - None: Unfreeze entire network

- **Supported Artistic Domains**
  - Watercolor2k: Watercolor paintings
  - Clipart1k: Clipart illustrations
  - Comic2k: Comic book artwork

## Dataset

The project uses COCO-style annotations but focuses on 6 VOC-style object categories:

| Category | COCO ID | Mapped ID |
|----------|---------|-----------|
| bicycle  | 1       | 0         |
| bird     | 2       | 1         |
| car      | 3       | 2         |
| cat      | 4       | 3         |
| dog      | 5       | 4         |
| person   | 6       | 5         |

### Dataset Download

```bash
bash prepare.sh
```

This script downloads all artistic datasets from Google Drive and organizes them in the correct directory structure.

## Project Structure

```
.
├── configs/              # Configuration management
│   └── configs.py       # FasterRCNNConfig class
├── datasets/            # Data handling
│   ├── dataloader.py    # PyTorch dataloader creation
│   ├── datasets.py      # VOCStyleDataset implementation
│   └── transforms.py    # Image transformations
├── engine/              # Training pipeline
│   ├── train_loop.py    # Trainer class
│   └── evaluator.py     # Evaluation metrics (mAP)
├── utils_/              # Utilities
│   ├── visualizer.py    # Detection visualization
│   ├── optimizer.py     # Optimizer configuration
│   └── utils.py         # General utilities
├── model.py             # Faster R-CNN model builder
├── main.py              # Entry point
└── prepare.sh           # Dataset download script
```

## Usage

### 1. Configuration

Edit the configuration in `main.py`:

```python
config = FasterRCNNConfig(
    dataset="watercolor",  # Options: watercolor, clipart, comic
    mode="fine-tune",      # Options: zero-shot, fine-tune, scratch
    freeze_backbone="none", # Options: full, partial, bn_only, none
    num_epochs=30,
    batch_size=16,
    learning_rate=0.0001
)
```

### 2. Training

```bash
python main.py
```

### 3. Results

- **Checkpoints**: `./checkpoints/{dataset}/{mode}/`
- **Visualizations**: `./visualization/{dataset}/{mode}/`

## Model Architecture

- **Backbone**: ResNet-50 with Feature Pyramid Network (FPN)
- **Detection Head**: Faster R-CNN
- **Input Size**: 512×512 pixels
- **Pre-trained Weights**: COCO dataset (for fine-tune and zero-shot modes)

## Training Details

- **Optimizer**: Adam (lr=0.0001)
- **LR Scheduler**: StepLR (decay by 0.1 every 7 epochs)
- **Data Augmentation**: Random horizontal flip (training only)
- **Batch Size**: 16
- **Epochs**: 30 (default)

## Evaluation Metrics

The evaluation computes mean Average Precision (mAP) at three IoU thresholds:
- mAP@0.50
- mAP@0.75
- mAP@0.90

Results are computed using `torchmetrics.MeanAveragePrecision`.

## Requirements

- PyTorch
- torchvision
- torchmetrics
- PIL
- tqdm
- matplotlib

## Research Applications

This codebase is designed for research in:
- Domain adaptation for object detection
- Cross-domain generalization
- Artistic style transfer effects on detection
- Few-shot learning in artistic domains

## Report

The detailed research report is available at: [`report/Domain_adaptive_detection.pdf`](./report/Domain_adaptive_detection.pdf)
