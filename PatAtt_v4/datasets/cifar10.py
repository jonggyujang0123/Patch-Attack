"""
Cifar10 Dataloader implementation
Cifar100 Dataloader implementation
"""
import logging
import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torch.utils.data import Subset
logger = logging.getLogger()

def get_loader_cifar10(args, class_wise=False):
    if args.val:
        transform_train = transforms.Compose([
            transforms.RandomCrop((args.img_size, args.img_size), padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])
    else:
        transform_train = transforms.Compose([
            #  transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomCrop((args.img_size, args.img_size), padding=4),
            transforms.RandomHorizontalFlip(),
            #  transforms.TrivialAugmentWide(),
            #  transforms.Pad(4),
            #  transforms.RandomRotation(30),
            #  transforms.RandomResizedCrop((args.img_size, args.img_size), scale= (0.7 , 1.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            #  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])


    transform_val = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        #transforms.RandomResizedCrop((cfg.img_size, cfg.img_size), scale= (0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    transform_test = transform_val

    dataset_train = datasets.CIFAR10(root="../data",
                                     train=True,
                                     download=True,
                                     transform = transform_train,
                                     )
    dataset_val = datasets.CIFAR10(root="../data",
                                     train=False,
                                     download=True,
                                     transform = transform_val,
                                     )
    dataset_test = datasets.CIFAR10(root="../data",
                                     train=False,
                                     download=True,
                                     transform=transform_test,
                                     ) 
    print("Cifar10 dataset loaded")
    

    if class_wise:
        loaders = []
        for name, class_ind in dataset_train.class_to_idx.items():
            if name == 'N/A':
                continue
            dataset_train_subset_ind = Subset(dataset_train, np.where(np.array(dataset_train.targets) == class_ind)[0])
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
