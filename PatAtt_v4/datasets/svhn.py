import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler,  TensorDataset, Dataset, Subset

import torchvision.utils as v_utils
from torchvision import datasets, transforms
import numpy as np


def get_loader_svhn(args, class_wise = None):
    transform_train = transforms.Compose([
                                    transforms.Resize((args.img_size, args.img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    transform_val = transforms.Compose([
                                    transforms.Resize((args.img_size, args.img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    transform_test = transform_val
    

    dataset_train = datasets.SVHN('../data',
                                    split='train',
                                    download=True,
                                    transform=transform_train)
    dataset_val = datasets.SVHN('../data', 
                                    split = 'test',
                                    download=True,
                                    transform=transform_val)
    dataset_test = datasets.SVHN('../data',
                                    split = 'test',
                                    download=True,
                                    transform=transform_test)

    if class_wise:
        loaders = []
        for class_ind in range(10):
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



