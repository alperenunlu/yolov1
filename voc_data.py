import os

import torch
from config_parser import YOLOConfig
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.datasets import VOCDetection, wrap_dataset_for_transforms_v2
from torchvision.transforms import v2
from yolo_utils import xyxy_to_yolo_target

n_cpu = os.cpu_count() or 0


class VOCDataModule:
    def __init__(self, config: YOLOConfig):
        self.config = config
        self.download = not os.path.exists("./data/VOCdevkit")
        self.train_transforms = self._train_transform_fn
        self.valid_transforms = self._valid_transform_fn

    def _train_transform_fn(self, image, target):
        v2_compose = v2.Compose(
            [
                v2.ToImage(),
                v2.RandomResizedCrop(
                    self.config.IMAGE_SIZE,
                    scale=(1 - self.config.Jitter, 1 + self.config.Jitter),
                    ratio=(1 - self.config.Jitter, 1 + self.config.Jitter),
                ),
                v2.RandomHorizontalFlip(),
                v2.RandomPhotometricDistort(),
                v2.ToDtype(torch.float32, scale=True),
                v2.ClampBoundingBoxes(),
                v2.SanitizeBoundingBoxes(),
            ]
        )
        image, target = v2_compose(image, target)
        yolo_target = xyxy_to_yolo_target(
            target["boxes"], target["labels"], self.config
        )

        annot = dict(
            id=target["annotation"]["filename"],
            boxes=target["boxes"],
            labels=target["labels"],
            difficult=[int(obj["difficult"]) for obj in target["annotation"]["object"]],
        )

        return image, yolo_target, annot

    def _valid_transform_fn(self, image, target):
        v2_compose = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(self.config.IMAGE_SIZE),
                v2.ToDtype(torch.float32, scale=True),
                v2.ClampBoundingBoxes(),
                v2.SanitizeBoundingBoxes(),
            ]
        )
        image, target = v2_compose(image, target)
        yolo_target = xyxy_to_yolo_target(
            target["boxes"], target["labels"], self.config
        )

        annot = dict(
            id=target["annotation"]["filename"],
            boxes=target["boxes"],
            labels=target["labels"],
            difficult=torch.tensor(
                [int(obj["difficult"]) for obj in target["annotation"]["object"]],
                dtype=torch.bool,
            ),
        )

        return image, yolo_target, annot

    def _collate_fn(self, batch):
        """Collate function to handle batching of images and targets."""
        images, yolo_targets, annot = zip(*batch)
        images = torch.stack(images, dim=0)
        yolo_targets = torch.stack(yolo_targets, dim=0)
        annot = [d for d in annot]
        return images, yolo_targets, annot

    def get_dataloaders(self) -> tuple[DataLoader, DataLoader]:
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
                ),
                target_keys="all",
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
            ),
            target_keys="all",
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
