import torch

class FasterRCNNConfig:
    def __init__(self, mode="finetune", dataset="watercolor"):
        self.seed = 42
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dataset Settings
        self.dataset = dataset
        self.image_size = (512, 512)
        self.num_workers = 4
        self.batch_size = 16
        self.num_classes = 6

        # Training Settings
        self.epoch = 30
        self.lr = 0.0001
        self.weight_decay=1e-4
        self.optimizer='Adam'


        # Model Settings
        self.model = 'fasterrcnn_resnet50_fpn'
        self.pretrained = True
        self.backbone_freeze = True
        self.head_random_init = False

        # Paths
        self.data_root = f'./data/{dataset}'
        self.output_dir = f'./checkpoints/{dataset}/{mode}'
        self.save_path = f'./visualization/{dataset}/{mode}'

        # Eval / Mode
        self.eval_only = False
        self.mode = mode  # "zeroshot" | "finetune" | "scratch"
        self.freeze_mode = "full"  # Options: "full", "partial", "none", "bn_only"
        self.configure_mode()


        # Dataset Information
        self.CLASSES = ['bicycle', 'bird', 'car', 'cat', 'dog', 'person']

    def configure_mode(self):
        if self.mode == "zeroshot":
            self.eval_only = True
            self.backbone_freeze = True
            self.head_random_init = False

        elif self.mode == "finetune":
            self.eval_only = False
            self.backbone_freeze = True
            self.head_random_init = False

            self.output_dir = f'./checkpoints/{self.dataset}/{self.mode}/{self.freeze_mode}'
            self.save_path = f'./visualization/{self.dataset}/{self.mode}/{self.freeze_mode}'

        elif self.mode == "scratch":
            self.eval_only = False
            self.backbone_freeze = True
            self.head_random_init = True

        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        