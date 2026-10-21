from __future__ import annotations

from pathlib import Path

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from .constants import CIFAR10_MEAN, CIFAR10_STD, Config


def train_transform():
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def test_transform():
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    )


def get_cifar10_testset(data_dir: str | Path, download: bool = False, transform=True):
    return torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=download,
        transform=test_transform() if transform else None,
    )


def get_dataloaders(config: Config, download: bool = True):
    root = config.resolved_data_dir()
    trainset = torchvision.datasets.CIFAR10(
        root=str(root), train=True, download=download, transform=train_transform()
    )
    testset = get_cifar10_testset(root, download=download)
    pin_memory = config.resolved_device().type == "cuda"
    return (
        DataLoader(
            trainset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
        ),
        DataLoader(
            testset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        ),
    )
