# VOC-Style Object Detection with Faster R-CNN

This repository provides configurable piple line for training and evaluating Faster R-CNN on COCO-sytle datasets with a subset of VOC-style classes (bicycle, bird, car, cat, dog, person).
It support multiple training mode (zero-shot, fine-tuning, and scratch), dataset variants (Watercolor2k, Clipart, Comic), and visualizations.

---

## Dataset
The project use COCO-style annotations, but only considers the following 6 VOC-style categories:

| COCO ID | Label   | Mapped ID |
|---------|---------|-----------|
| 1       | bicycle | 0         |
| 2       | bird    | 1         |
| 3       | car     | 2         |
| 4       | cat     | 3         |
| 5       | dog     | 4         |
| 6       | person  | 5         |

**Supported datasets**:
- [x] Watercolor2k
- [x] Clipart1k
- [x] Comic2k

You can download dataset with:
```
bash prepare.sh
```

---

## Project Structure
```
.
├── config/               
├── data/
│   ├── clipart/
│   ├── watercolor/
│   └── comic/
├── datasets/
│   ├── dataloader.py
│   ├── datasets.py
│   └── transforms.py
├── engine/
│   ├── train_loop.py
│   └── evaluator.py
├── utils/
│   ├── convertToCOCO.py
│   ├── optimizer.py
│   ├── utils.py
│   └── visualizer.py
├── model.py
└── main.py 
```
---

## Training
python main.py

---

## Evaluation
Evaluation uses torchmetrics.MeanAveragePrecision with:

* mAP@0.50
* mAP@0.75
* mAP@0.90

Checkpoints are saved under:
```
./checkpoints/{dataset}/{mode}/
```
