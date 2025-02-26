import os

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import ConcatDataset

from torchvision.datasets import VOCDetection
from torchvision.datasets import wrap_dataset_for_transforms_v2
from torchvision.transforms import v2

from yolo_utils import xyxy_to_yolo_target

from typing import Tuple

from config_parser import YOLOConfig

n_cpu = os.cpu_count() or 0
download = os.path.exists("./data/VOCdevkit")


class VOCDataModule:
    def __init__(self, config: YOLOConfig):
        self.config = config
        self.download = not __import__("os").path.exists("./data/VOCdevkit")
        self.train_transforms = self._get_train_transforms()
        self.valid_transforms = self._get_valid_transforms()

    def _get_train_transforms(self):
        return v2.Compose(
            [
                v2.ToImage(),
                v2.RandomHorizontalFlip(),
                v2.ColorJitter(brightness=0.5, contrast=0, saturation=0.5, hue=0),
                v2.RandomAffine(
                    degrees=0,
                    scale=(0.8, 1.2),
                    translate=(0.2, 0.2),
                    shear=0,
                ),
                v2.Resize(self.config.IMAGE_SIZE),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                v2.ClampBoundingBoxes(),
                v2.SanitizeBoundingBoxes(),
            ]
        )

    def _get_valid_transforms(self):
        """Returns a transform pipeline for validation."""
        return v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(self.config.IMAGE_SIZE),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                v2.ClampBoundingBoxes(),
                v2.SanitizeBoundingBoxes(),
            ]
        )

    def _collate_fn(self, batch):
        """Collate function to handle batching of images and targets."""
        images, targets = zip(*batch)
        images = torch.stack(images)
        yolo_targets = torch.stack(
            [
                xyxy_to_yolo_target(target["boxes"], target["labels"], self.config)
                for target in targets
            ]
        )
        return images, yolo_targets, targets

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Returns training and validation DataLoaders.
        """
        train_datasets = [
            wrap_dataset_for_transforms_v2(
                VOCDetection(
                    root="./data",
                    year=year,
                    image_set=split,
                    download=self.download,
                    transforms=self.train_transforms,
                )
            )
            for year in ["2007", "2012"]
            for split in ["train", "val"]
        ]

        valid_datasets = wrap_dataset_for_transforms_v2(
            VOCDetection(
                root="./data",
                year="2007",
                image_set="test",
                download=self.download,
                transforms=self.valid_transforms,
            )
        )

        train_datasets_concat: Dataset = ConcatDataset(train_datasets)

        train_loader = DataLoader(
            train_datasets_concat,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            collate_fn=self._collate_fn,
            num_workers=n_cpu,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        valid_loader = DataLoader(
            valid_datasets,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            collate_fn=self._collate_fn,
            num_workers=n_cpu,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        return train_loader, valid_loader


if __name__ == "__main__":
    from config_parser import load_config

    config = load_config("yolo_config.yaml")

    from yolo_utils import yolo_target_to_xyxy

    voc_data = VOCDataModule(config)

    train_loader, valid_loader = voc_data.get_dataloaders()
    print(len(train_loader), len(valid_loader))
    it = iter(train_loader)
    images, yolo_labels, labels = next(it)
    print(images.shape, yolo_labels.shape)
    print(labels[0])
    print(yolo_target_to_xyxy(yolo_labels, config)[0])
