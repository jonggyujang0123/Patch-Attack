
"""
LFW Dataloader implementation, used in DCGAN
"""
import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler,  TensorDataset, Dataset, Subset
import torchvision.utils as v_utils
from torchvision import datasets, transforms
import numpy as np


def get_loader_LFW(args, class_wise=False):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        #  transforms.CenterCrop(128),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    dataset_train = datasets.LFWPeople(
        root='../data',
        split='train',
        image_set='deepfunneled',
        transform=transform,
        download=True
        )
    dataset_test = datasets.LFWPeople(
        root='../data',
        split='test',
        image_set='deepfunneled',
        transform=transform,
        download=True
        )
    if class_wise:
        dataset_train.targets_np = np.array(dataset_train.targets)
        loaders = []
        for name, class_ind in enumerate([291, 304, 321, 370, 373, 380, 385, 417, 531, 538]):
            print(class_ind)
            if name == 'N/A':
                continue
            dataset_train_subset_ind = Subset(dataset_train, np.where(dataset_train.targets_np == class_ind)[0])
            loader = DataLoader(
                dataset=dataset_train_subset_ind,
                batch_size=args.test_batch_size,
                num_workers=args.num_workers,
                shuffle=False,
                pin_memory=True
                )
            loaders.append(loader)
        return loaders, 0, 0
    else:
        train_loader = DataLoader(
            dataset=dataset_train,
            #  sampler=RandomSampler(dataset_train),
            batch_size=args.train_batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=True
            )
        test_loader = DataLoader(
            dataset=dataset_test,
            #  sampler=SequentialSampler(dataset_test),
            batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=True
            )
        return train_loader, test_loader, test_loader


