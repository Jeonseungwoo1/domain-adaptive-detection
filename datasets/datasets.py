from torchvision.datasets import CocoDetection
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import torchvision.transforms as T
from PIL import Image
import os
import torch

CATEGORY_ID_TO_LABEL = {
    1: 0,   # bicycle
    2: 1,   # bird
    3: 2,   # car
    4: 3,   # cat
    5: 4,  # dog
    6: 5   # person
}
class VOCStyleDataset(CocoDetection):
    """
    A customized COCO-format dataset that uses PASCAL VOC-style categories (bicycle, bird, car, cat, dog, person).
    This dataset is suitable for experiments in domain adaptation, fine-tuing, and zero-shot, detection on artistic datasets
    suchas Watercolor2k, Comic, and Clipart.

    Args:
        img_folder (str): Path to the folder containing images.
        ann_file (str): Path to the COCO-format annotation JSON file.
        transforms (callable, optional): A function/transform to apply to the input image.

    Return:
        tuple: A tuple (image, target):
            - imgae (Tensor): The transformed image tensor of shape (C, H, W).
            - target (dict): A dictionary with keys:
                - 'boxes' (Tensor): Bounding boxed in [x1, y1, x2, y2] format, shape (N, 4).
                - 'labels' (Tensor): Class labels (0~5) correseponding to VOC-style categories.

    Notes:
        - If an image contains no valid annotations from the VOC subset, it is skipped and the next image is used.
        - Bounding boxes are rescaled automatically based on image resizing.
        - Assumes the COCO-style annotation contains VOC-style category IDs from 1 to 6 only.
    """
    def __init__(self, img_folder, ann_file, transforms=None):
        super(VOCStyleDataset, self).__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        while True:
            img, _ = super().__getitem__(idx)
            img_id = self.ids[idx]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            boxes = []
            labels = []

            for ann in anns:
                cat_id = ann['category_id']
                if cat_id not in CATEGORY_ID_TO_LABEL:
                    continue
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x + w, y + h])
                labels.append(CATEGORY_ID_TO_LABEL[cat_id])

            if len(boxes) == 0:
                idx = (idx + 1) % len(self)
                continue

            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

            orig_w, orig_h = img.size
            if self._transforms is not None:
                img = self._transforms(img)

            new_h, new_w = img.shape[1:]  # (C, H, W)
            scale_x = new_w / orig_w
            scale_y = new_h / orig_h
            boxes *= torch.tensor([scale_x, scale_y, scale_x, scale_y])

            target = {
                'boxes': boxes,
                'labels': labels
            }

            return img, target
    
    def __len__(self):
        return len(self.ids)
    
