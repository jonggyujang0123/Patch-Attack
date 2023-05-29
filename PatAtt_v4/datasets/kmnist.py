"""
KMNIST dataset loader
"""

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler,  TensorDataset, Dataset, Subset

import torchvision.utils as v_utils
from torchvision import datasets, transforms
import numpy as np


def get_loader_kmnist(args, class_wise=False):
    train_transform = transforms.Compose([
        transforms.Resize((30,30)),
        transforms.Pad(1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((30,30)),
        transforms.Pad(1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset_train = datasets.KMNIST(
        root='../data',
        download=True,
        train=True,
        transform=train_transform,
        )
    dataset_test = datasets.KMNIST(
        root='../data',
        download=True,
        train=False,
        transform=test_transform,
        )
    
    if class_wise:
        loaders = []
        for name, class_ind in dataset_train.class_to_idx.items():
            if name == 'N/A':
                continue
            dataset_train_subset_ind = Subset(dataset_train, np.where(dataset_train.targets == class_ind)[0])
            loader = DataLoader(dataset_train_subset_ind,
                                batch_size=args.test_batch_size,
                                shuffle=True,
                                num_workers=args.num_workers,
                                pin_memory = args.pin_memory)
            loaders.append(loader)
        return loaders, 0, 0
    else:
        train_loader = DataLoader(dataset_train,
                                  sampler=RandomSampler(dataset_train),
                                  batch_size = args.train_batch_size,
                                  num_workers = args.num_workers,
                                  pin_memory=args.pin_memory)
        test_loader = DataLoader(dataset_test,
                                  sampler=SequentialSampler(dataset_test),
                                  batch_size = args.test_batch_size,
                                  num_workers = args.num_workers,
                                  pin_memory=args.pin_memory)
        return train_loader, test_loader, test_loader

