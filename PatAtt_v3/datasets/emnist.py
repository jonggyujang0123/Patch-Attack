"""
Mnist Data loader, as given in Mnist tutorial
"""
import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler,  TensorDataset, Dataset

import torchvision.utils as v_utils
from torchvision import datasets, transforms



def get_loader_emnist(args):
    transform_train = transforms.Compose([
#                                    transforms.RandomHorizontalFlip(p=.5),
#                                    transforms.RandomVerticalFlip(p=.5),
#                                    transforms.RandomRotation(23),
#                                    transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale = (1.0, 1.2)),
#                                    transforms.ColorJitter(
#                                        brightness = (0.7, 2),
#                                        contrast=(0.7, 2),
#                                        ),
#                                    transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
#                                    transforms.RandomInvert(0.1),
                                    lambda img: transforms.functional.rotate(img, -90),
                                    lambda img: transforms.functional.hflip(img),
                                    transforms.Resize((args.img_size, args.img_size)),
#                                    transforms.AugMix(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    transform_val = transforms.Compose([
                                    lambda img: transforms.functional.rotate(img, -90),
                                    lambda img: transforms.functional.hflip(img),
                                    transforms.Resize((args.img_size, args.img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    transform_test = transform_val

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
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

    #dataset_train, dataset_val = torch.utils.data.random_split(dataset, [50000, 10000], generator=torch.Generator().manual_seed(1))
    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(dataset_train) if args.local_rank == -1 else DistributedSampler(dataset_train)
    val_sampler = RandomSampler(dataset_val) if args.local_rank == -1 else DistributedSampler(dataset_val)
#    val_sampler = SequentialSampler(dataset_val)
    test_sampler = RandomSampler(dataset_test) if args.local_rank == -1 else DistributedSampler(dataset_test)
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
