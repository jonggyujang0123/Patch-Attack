"""
Omniglot dataset loader
"""

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler,  TensorDataset, Dataset

import torchvision.utils as v_utils
from torchvision import datasets, transforms


def get_loader_omniglot(args):
    train_transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset_train = datasets.Omniglot(
        root='../data',
        download=True,
        background=True,
        transform=train_transform,
        )
    dataset_test = datasets.Omniglot(
        root='../data',
        download=True,
        background=False,
        transform=test_transform,
        )
    train_loader = DataLoader(
        dataset_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=RandomSampler(dataset_train),
        )
    test_loader = DataLoader(
        dataset_test,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=SequentialSampler(dataset_test),
        )
    return train_loader, test_loader, test_loader
