from torch.utils.data import DataLoader
from .transforms import get_transform
from .datasets import VOCStyleDataset
from utils_.utils import collate_fn

def get_dataloader(config, split):

    ann_path = f"{config.data_root}/annotations/{config.dataset}_{split}.json"
    img_dir = f"{config.data_root}/JPEGImages"

    is_train = split == "train"

    dataset = VOCStyleDataset(
        img_folder=img_dir,
        ann_file=ann_path,
        transforms=get_transform(train=is_train)
    )

    dataloader = DataLoader(
        dataset,
        batch_size = config.batch_size,
        shuffle=is_train,
        num_workers=config.num_workers,
        collate_fn = collate_fn
    )

    return dataloader

