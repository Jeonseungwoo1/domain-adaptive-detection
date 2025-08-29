import torchvision
import torch
import os
import random
import numpy as np
import pandas as pd
import datetime
import wandb
from configs.configs import FasterRCNNConfig
from datasets.datasets import VOCStyleDataset
from model import build_faster_rcnn
from datasets.dataloader import get_dataloader
from utils_.utils import set_seed
from engine.train_loop import Trainer
from engine.evaluator import Evaluator
from utils_.optimizer import get_optimizer
from utils_.visualizer import visualize_prediction, visualize_ground_truth_and_predictions
from datasets.transforms import get_transform



def main():
    config = FasterRCNNConfig(mode = "finetune", dataset= "watercolor") # Mode options :| "zeroshot" | "finetune" | "scratch" |
    set_seed(config.seed)                                               # Dataset Options: | "clipart" | "watercolor" | "comic" | 

    # logger = wandb.init(
    #              project="DL-termproject",
    #              name = f"{config.model}_{config.mode}_{config.dataset}",
    #              entity='',
    #              config=config.__dict__
    #             )
    logger = None

    # Initialize model
    model = build_faster_rcnn(config).to(config.device)

    ann_path = f"{config.data_root}/annotations/{config.dataset}_test.json"
    img_dir = f"{config.data_root}/JPEGImages"

    dataset = VOCStyleDataset(
        img_folder=img_dir,
        ann_file=ann_path,
        transforms=get_transform(train=False)
    )

    # Load dataloader
    train_loader = get_dataloader(config, "train")
    test_loader = get_dataloader(config, "test")


    # Initialize Training Loop
    if config.eval_only:
        print(f"[{config.mode.upper()}] Evaluation only (no training).")
        if config.mode != 'zeroshot':
            model.load_state_dict(torch.load(f"{config.output_dir}/best_model.pth"))
        evaluator = Evaluator(model, test_loader, config.device, logger)
        evaluator.evaluate()
    else:
        print(f"[{config.mode.upper()}] Training mode.")
        train_loader = get_dataloader(config, split = "train")
        optimizer = get_optimizer(config, model)
        trainer = Trainer(model, optimizer, train_loader, test_loader, config.device, logger)
        trainer.train(epochs=config.epoch, save_path=config.output_dir)

    if config.mode != 'zeroshot':
        models = []
        model.load_state_dict(torch.load(f"{config.output_dir}/best_model.pth"))
        models.append(model)
    else:
        models = []
        model.load_state_dict(torch.load(f"{config.output_dir}/best_model.pth"))
        models.append(model)
    visualize_ground_truth_and_predictions(models, "P", dataset, config.device, config, target_class="bird", start_idx=200, score_threshold=0.4, save_path=config.save_path)
        
if __name__ == '__main__':
    main()