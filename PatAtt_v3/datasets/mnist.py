"""
Mnist Data loader, as given in Mnist tutorial
"""
import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler,  TensorDataset, Dataset

import torchvision.utils as v_utils
from torchvision import datasets, transforms



def get_loader_mnist(args):
    transform_train = transforms.Compose([
                                    transforms.Resize((args.img_size, args.img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    transform_val = transforms.Compose([
                                    transforms.Resize((args.img_size, args.img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    transform_test = transform_val

    dataset_train = datasets.MNIST('../data', 
                                    train=True, 
                                    download=True,
                                    transform=transform_train)
    dataset_val = datasets.MNIST('../data', 
                                    train=False, 
                                    download=True,
                                    transform=transform_val)
    dataset_test = datasets.MNIST('../data', 
                                    train=False, 
                                    download=True,
                                    transform=transform_test)

    #dataset_train, dataset_val = torch.utils.data.random_split(dataset, [50000, 10000], generator=torch.Generator().manual_seed(1))

    train_sampler = RandomSampler(dataset_train)
    val_sampler = RandomSampler(dataset_val)
#    val_sampler = SequentialSampler(dataset_val)
    test_sampler = RandomSampler(dataset_test)
    train_loader = DataLoader(dataset_train,
                              sampler=train_sampler,
                              batch_size = args.train_batch_size,
                              num_workers = args.num_workers,
                              pin_memory=args.pin_memory)
    val_loader = DataLoader(dataset_val,
                              sampler=val_sampler,
                              batch_size = args.train_batch_size,
                              num_workers = args.num_workers,
                              pin_memory=args.pin_memory) 
    test_loader = DataLoader(dataset_test,
                              sampler=test_sampler,
                              batch_size = args.test_batch_size,
                              num_workers = args.num_workers,
                              pin_memory=args.pin_memory)
    return train_loader, val_loader, test_loader
