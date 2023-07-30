"""
Mnist Data loader, as given in Mnist tutorial
"""
import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler,  TensorDataset, Dataset, Subset

import torchvision.utils as v_utils
from torchvision import datasets, transforms
import numpy as np


def get_loader_emnist(args, class_wise = None):
    transform_train = transforms.Compose([
                                    lambda img: transforms.functional.rotate(img, -90),
                                    lambda img: transforms.functional.hflip(img),
                                    #  transforms.RandomRotation(20),
                                    #  transforms.Pad(4),
                                    #  transforms.RandomResizedCrop(args.img_size, scale=(0.7, 1.0), ratio=(1.0,1.0)),
                                    transforms.Resize((args.img_size, args.img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    transform_val = transforms.Compose([
                                    lambda img: transforms.functional.rotate(img, -90),
                                    lambda img: transforms.functional.hflip(img),
                                    transforms.Resize((args.img_size, args.img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    transform_test = transform_val
    

    dataset_train = datasets.EMNIST('../data',
                                    split= 'letters',
                                    train=True, 
                                    download=True,
                                    transform=transform_train)
    dataset_val = datasets.EMNIST('../data', 
                                    split ='letters',
                                    train=False, 
                                    download=True,
                                    transform=transform_val)
    dataset_test = datasets.EMNIST('../data', 
                                    split = 'letters',
                                    train=False, 
                                    download=True,
                                    transform=transform_test)
    dataset_train.targets = dataset_train.targets - 1
    dataset_val.targets = dataset_val.targets - 1
    dataset_test.targets = dataset_test.targets - 1

    if class_wise:
        loaders = []
        for class_ind in range(26):
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
        val_loader = DataLoader(dataset_val,
                                  sampler=SequentialSampler(dataset_val),
                                  batch_size = args.train_batch_size,
                                  num_workers = args.num_workers,
                                  pin_memory=args.pin_memory) 
        test_loader = DataLoader(dataset_test,
                                  sampler=SequentialSampler(dataset_test),
                                  batch_size = args.test_batch_size,
                                  num_workers = args.num_workers,
                                  pin_memory=args.pin_memory)
        return train_loader, val_loader, test_loader
