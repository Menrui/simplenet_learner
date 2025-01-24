from pathlib import Path
from typing import Optional, Union

import torch
from lightning import LightningDataModule
from omegaconf import DictConfig
from torchvision import transforms

from simplenet_learner.datamodules.components.mvtecad import MVTecadDataset
from simplenet_learner.datamodules.components.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from simplenet_learner.utils import get_logger

logger = get_logger(__name__)


class MVTecadLitDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Union[str, Path],
        category: str,
        transform_cfg: DictConfig,
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super(MVTecadLitDataModule, self).__init__()
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        self.data_dir = data_dir
        self.category = category
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.transform_cfg = transform_cfg

        self.train_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.transform_cfg.resize_h, self.transform_cfg.resize_w)
                ),
                transforms.RandomHorizontalFlip(self.transform_cfg.h_flip_p),
                transforms.RandomVerticalFlip(self.transform_cfg.v_flip_p),
                transforms.CenterCrop(
                    (self.transform_cfg.crop_h, self.transform_cfg.crop_w)
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.transform_cfg.resize_h, self.transform_cfg.resize_w)
                ),
                transforms.CenterCrop(
                    (self.transform_cfg.crop_h, self.transform_cfg.crop_w)
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.transform_cfg.resize_h, self.transform_cfg.resize_w)
                ),
                transforms.CenterCrop(
                    (self.transform_cfg.crop_h, self.transform_cfg.crop_w)
                ),
                transforms.ToTensor(),
            ]
        )

    def prepare_data(self):
        # download
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            # Load the training dataset
            dataset = MVTecadDataset(
                root=str(self.data_dir),
                category=self.category,
                phase="train",
                transform=self.train_transform,
                mask_transform=self.mask_transform,
            )
            n_samples = len(dataset)
            train_size = int(n_samples * 0.8)
            val_size = n_samples - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
            logger.info(f"  - Training dataset size: {train_size}")
            logger.info(f"  - Validation dataset size: {val_size}")

        if stage == "test" or stage is None:
            # Load the test dataset
            self.test_dataset = MVTecadDataset(
                root=str(self.data_dir),
                category=self.category,
                phase="test",
                transform=self.test_transform,
                mask_transform=self.mask_transform,
            )
            logger.info(f"  - Test dataset size: {len(self.test_dataset)}")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
