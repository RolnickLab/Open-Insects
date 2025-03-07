from datasets import load_dataset
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# for BioCLIP validation ID
def download_inat21():
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )
    train_dataset = datasets.INaturalist(
        root="/network/scratch/y/yuyan.chen/inat21",
        version="2021_train",
        transform=transform,
        download=True,
    )

    print(len(train_dataset))

    train_dataset = datasets.INaturalist(
        root="/network/scratch/y/yuyan.chen/inat21",
        version="2021_valid",
        transform=transform,
        download=True,
    )

    print(len(train_dataset))


# for BioCLIP
