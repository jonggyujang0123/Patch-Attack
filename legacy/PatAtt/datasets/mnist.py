"""
Mnist Data loader, as given in Mnist tutorial
"""
import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler,  TensorDataset, Dataset

import torchvision.utils as v_utils
from torchvision import datasets, transforms



def get_loader_mnist(cfg, args):
    transform_train = transforms.Compose([
                                    transforms.Resize((cfg.img_size, cfg.img_size)),
                                    transforms.ToTensor()])
    transform_val = transforms.Compose([
                                    transforms.Resize((cfg.img_size, cfg.img_size)),
                                    transforms.ToTensor()])
    transform_test = transform_val

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
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
    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(dataset_train) if args.local_rank == -1 else DistributedSampler(dataset_train)
    val_sampler = RandomSampler(dataset_val) if args.local_rank == -1 else DistributedSampler(dataset_val)
#    val_sampler = SequentialSampler(dataset_val)
    test_sampler = RandomSampler(dataset_test) if args.local_rank == -1 else DistributedSampler(dataset_test)
    train_loader = DataLoader(dataset_train,
                              sampler=train_sampler,
                              batch_size = cfg.train_batch_size,
                              num_workers = cfg.num_workers,
                              pin_memory=cfg.pin_memory)
    val_loader = DataLoader(dataset_val,
                              sampler=val_sampler,
                              batch_size = cfg.train_batch_size,
                              num_workers = cfg.num_workers,
                              pin_memory=cfg.pin_memory) 
    test_loader = DataLoader(dataset_test,
                              sampler=test_sampler,
                              batch_size = cfg.test_batch_size,
                              num_workers = cfg.num_workers,
                              pin_memory=cfg.pin_memory)
    return train_loader, val_loader, test_loader
