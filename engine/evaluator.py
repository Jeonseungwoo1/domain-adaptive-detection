import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

class Evaluator:
    """
    Evaluator for computing mAP for object detection models.
    
    Attributes:
        model : The model to be evaluated.
        dataloader: Dataloader containing evaluation data.
        logger: Logger object (e.g., wandb) to record metrics
        mAP_50: Metric object for IoU@0.50.
        mAP_75: Metric object for IoU@0.75.
        mAP_90: Metric object for IoU@0.90.
    """
    def __init__(self, model, dataloader, device, logger=None):
        """
        Initialize the Evaluator.

        Args:
            model (torch.nn.Module)
            dataloader (Dataloader)
        """
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.logger = logger

        self.mAP_50 = MeanAveragePrecision(iou_thresholds=[0.5])
        self.mAP_75 = MeanAveragePrecision(iou_thresholds=[0.75])
        self.mAP_90 = MeanAveragePrecision(iou_thresholds=[0.9])


    def evaluate(self, epoch=None):
        """
        Perform model evaluation on the entire evaluation dataset.
        
        Args:
            epoch (int, optional): Current epoch index (for logging)
        
        Return:
            map_50 (float): mAP@0.50 score
        """
        self.model.eval()
        self.mAP_50.reset()
        self.mAP_75.reset()
        self.mAP_90.reset()

        pbar = tqdm(self.dataloader, desc="Evaluating")
        with torch.no_grad():
            for images, targets in pbar:
                if len(images) == 0:
                    continue
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to("cpu") for k, v in t.items()} for t in targets]
                outputs = self.model(images)
                outputs = [{k: v.to("cpu") for k, v in o.items()} for o in outputs]

                self.mAP_50.update(outputs, targets)
                self.mAP_75.update(outputs, targets)
                self.mAP_90.update(outputs, targets) 
        
            result_50 = self.mAP_50.compute()
            result_75 = self.mAP_75.compute()
            result_90 = self.mAP_90.compute()

            map_50 = result_50['map'].item()
            map_75 = result_75['map'].item()
            map_90 = result_90['map'].item()

            print("========== Evaluation Result ==========")
            print(f"    mAP@0.50: {map_50:.8f}")
            print(f"    mAP@0.75: {map_75:.8f}")
            print(f"    mAP@0.90: {map_90:.8f}")

            if self.logger:
                self.logger.log({
                    "val/mAP@0.50": map_50,
                    "val/mAP@0.75": map_75,
                    "val/mAP@0.90": map_90,
                    "epoch": epoch
                })
            
            return map_50
