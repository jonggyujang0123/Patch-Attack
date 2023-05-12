
"""
LFW Dataloader implementation, used in DCGAN
"""
import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler,  TensorDataset, Dataset

import torchvision.utils as v_utils
from torchvision import datasets, transforms
import numpy as np


def get_loader_lfw(args):
    transform = transforms.Compose([
       transforms.Resize(64),
       transforms.ToTensor(),
       transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    dataset_train = datasets.LFWPeople(
        root='../data',
        split='train',
        transform=transform,
        download=True
        )
    dataset_test = datasets.LFWPeople(
        root='../data',
        split='test',
        transform=transform,
        download=True
        )
    train_loader = DataLoader(
        dataset=dataset_train,
        sampler=RandomSampler(dataset_train),
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True
        )
    test_loader = DataLoader(
        dataset=dataset_test,
        sampler=SequentialSampler(dataset_test),
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True
        )
    return train_loader, test_loader, test_loader


