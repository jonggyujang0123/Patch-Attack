"""
CelebA Dataloader implementation, used in DCGAN
"""
import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler,  TensorDataset, Dataset, Subset
import numpy as np
import torchvision.utils as v_utils
from torchvision import datasets, transforms
import numpy as np
import torch.nn.functional as F
#  import albumentations as A
#  from albumentations.pytorch import ToTensorV2


def get_loader_celeba(args, class_wise=False):
    transform = transforms.Compose([
        #  transforms.RandomRotation(45, fill=0, expand=True),
        transforms.CenterCrop(148),
        #  transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9,1.1)),
        #  transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
        #  Cutout(n_holes=3, length=16),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    transform_test = transforms.Compose([
        transforms.CenterCrop(148),
        #  transforms.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(1.0,1.0)),
        #  transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        #  transforms.CenterCrop(128),
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    dataset_train = datasets.CelebA(
        root='../data',
        split='all',
        target_type='identity',
        transform=transform,
        download=True)
    dataset_test = datasets.CelebA(
        root='../data',
        split='all',
        target_type='identity',
        transform=transform_test,
        download=True)
    #  dataset_test = datasets.CelebA(
    #      root='../data',
    #      split='test',
    #      target_type='identity',
    #      transform=transform,
    #      download=True)
    #  dataset_val = datasets.CelebA(
    #      root='../data',
    #      split='valid',
    #      target_type='identity',
    #      transform=transform,
    #      download=True)
    dataset_train.identity = dataset_train.identity - 1
    dataset_test.identity = dataset_test.identity - 1
    #  dataset_val.identity = dataset_val.identity - 1

    train_targets = F.one_hot(dataset_train.identity, num_classes= 10177)
    indices = torch.argsort(train_targets.sum(1).sum(0), descending=True).view(-1)
    train_targets = train_targets[:, :, indices]
    dataset_train.identity = train_targets.sum(dim=1).argmax(dim=1).unsqueeze(1)
    
    test_targets = F.one_hot(dataset_test.identity, num_classes= 10177)
    test_targets = test_targets[:, :, indices]
    dataset_test.identity = test_targets.sum(dim=1).argmax(dim=1).unsqueeze(1)
    #
    #  val_targets = F.one_hot(dataset_val.identity, num_classes= 10177)
    #  val_targets = val_targets[:, :, indices]
    #  dataset_val.identity = val_targets.sum(dim=1).argmax(dim=1).unsqueeze(1)


    if class_wise:
        loaders = []
        for class_ind in range(args.num_classes):
            dataset_train_subset_ind = Subset(dataset_test, np.where(dataset_test.identity[:,0] == class_ind)[0])
            loader = DataLoader(dataset_train_subset_ind,
                                batch_size=args.test_batch_size,
                                shuffle=True,
                                num_workers=args.num_workers,
                                pin_memory = args.pin_memory)
            loaders.append(loader)
        return loaders, 0, 0
    else:
        n_total = len(dataset_train)
        n_target = int(n_total * 0.8)
        #  n_val = n_total - n_target
        mask = torch.zeros(n_total).bool()
        mask_ind = torch.randperm(n_total)[:n_target]
        mask[mask_ind] = True
        dataset_train = Subset(dataset_train, 
                               np.where(mask * (dataset_train.identity[:,0] < args.num_classes))[0])
        dataset_test = Subset(dataset_test,
                              np.where( (~mask) * (dataset_test.identity[:,0] < args.num_classes))[0])
        train_loader = DataLoader(
            dataset_train,
            #  sampler=RandomSampler(dataset_train),
            batch_size=args.train_batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=True)
        test_loader = DataLoader(
            dataset_test,
            #  sampler=SequentialSampler(dataset_test),
            batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False)
        return train_loader, test_loader, test_loader




