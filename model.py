import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_faster_rcnn(config):
    """
    Build a Faster R-CNN model based on ResNet-50 FPN backbone

    Args:
        config: Congfiguration object with the following attributes:

            - pretrained (bool): Whether to load COCO pretrained weights.
            - backbone_freeze (bool): Whether to freeze the backbone layers.
            - head_random_init (bool): Whether to randomly initialize the detection head. (for trian head on scratch)
            - num_classes (int): Number of object classes (excluding background).
            - image_size (tuple or int): Image size for min_size/max_size if needed.

    Return:
        model: A customized Faster R-CNN model.
    """

    # Load pretrained weights 
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if config.pretrained else None
    if config.mode == 'scratch':
        print("random backbone weight")
        model = fasterrcnn_resnet50_fpn(
            min_size=512, 
            max_size=512, 
            weights=None,
            weights_backbone = None
        )
    else:
        model = fasterrcnn_resnet50_fpn(
            min_size=512, 
            max_size=512, 
            weights=weights
        )

    # Backbone Freezing based on freeze_mode
    if config.freeze_mode == "full":
        for param in model.backbone.parameters():
            param.requires_grad = False

    elif config.freeze_mode == "partial":
        for name, param in model.backbone.body.named_parameters():
            if "layer4" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    elif config.freeze_mode == "bn_only":
        for name, param in model.backbone.body.named_parameters():
            if "bn" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    elif config.freeze_mode == "none":
        for param in model.backbone.parameters():
            param.requires_grad = True

    else:
        raise ValueError(f"Invalid freeze_mode: {config.freeze_mode}")


    # Replace head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, config.num_classes)

    # Random initialize head — for semi-scratch
    if config.head_random_init:
        torch.nn.init.normal_(model.roi_heads.box_predictor.cls_score.weight, std=0.01)
        torch.nn.init.constant_(model.roi_heads.box_predictor.cls_score.bias, 0)
        torch.nn.init.normal_(model.roi_heads.box_predictor.bbox_pred.weight, std=0.001)
        torch.nn.init.constant_(model.roi_heads.box_predictor.bbox_pred.bias, 0)

    return model


