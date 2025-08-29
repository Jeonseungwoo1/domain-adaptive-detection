import torch
from tqdm import tqdm
from engine.evaluator import Evaluator
from torch.optim.lr_scheduler import StepLR
import os

class Trainer:
    """
    Triner class for training and evaluating a mode

    Attributes:
        model : The object detection model.
        optimizer: The optimizer for training.
        train_loader: Dataloader for training dataset.
        test_loader: Dataloader for evaluation.
        logger: Logger object (e.g., wandb) to record metrics
        lr_schedular: Learning rate schedular (step decay)
        early_stopping: Number of epochs to wait before early stopping if no improvement on performance
        evaluator: Evaluator object for computing mAP after each epoch.
    """
    def __init__(self, model, optimizer, train_loader, test_loader, device, logger=None):
        """
        Initialize the Trainer.
        """
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.logger = logger
        self.lr_schedular = StepLR(optimizer, step_size=7, gamma=0.1)

        self.evaluator = Evaluator(self.model, self.test_loader, self.device, self.logger)

    def train(self, epochs, save_path=None):
        """
        Perform training and evaluation for a given number of epochs.
        
        Args:
            epochs (int): Total number of training epochs.
            save_path (str): Directory to save checkpoints
        
        """
        best_score = -1.0

        for epoch in range(epochs):
            loss = self.train_one_epoch(epoch)

            if self.lr_schedular is not None:
                self.lr_schedular.step()
            
            if self.evaluator:
                score = self.evaluator.evaluate(epoch)

                if score > best_score:
                    best_score = score
                    if save_path:
                        os.makedirs(save_path, exist_ok=True)
                        torch.save(self.model.state_dict(), f"{save_path}/best_model.pth")
                        print(f"[Checkpoint] Best model saved (score: {score:.4f})")
            
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                torch.save(self.model.state_dict(), f"{save_path}/model_epoch_{epoch}.pth")
            
    def train_one_epoch(self, epoch):
        """
        Train the model for one epoch.
        
        Args:
            epoch (int): Current epoch index.
            
        Returns:
            avg_loss (float): Average loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"[Train] Epoch {epoch}")
        for images, targets in pbar:
            if len(images) == 0:
                continue
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if self.logger:
                self.logger.log({
                    "train/loss_step": loss.item(),
                    "train/loss_classifier": loss_dict["loss_classifier"].item(),
                    "train/loss_box_reg": loss_dict["loss_box_reg"].item(),
                    "train/loss_objectness": loss_dict["loss_objectness"].item(),
                    "train/loss_rpn_box_reg": loss_dict["loss_rpn_box_reg"].item(),
                })

        avg_loss = total_loss / len(self.train_loader)
        print(f"[Train] Avg Loss: {avg_loss:.4f}")

        if self.logger:
            self.logger.log({
                "train/loss_epoch": avg_loss,
                "epoch": epoch
            })

        return avg_loss
