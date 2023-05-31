"""
Mnist Data loader, as given in Mnist tutorial
"""

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler,  TensorDataset, Dataset, Subset

import torchvision.utils as v_utils
from torchvision import datasets, transforms
import numpy as np
import torch.nn.functional as F


class SquarePad(object):
    def __call__(self, image):
        _, width, height = image.size()
        max_dim = max(width, height)
        p_left, p_top = [(max_dim - s) // 2 for s in image.size()[1:]]
        p_right, p_bottom = [max_dim - (s+pad) for s, pad in zip(image.size()[1:], [p_left, p_top])]
        padding = (p_left, p_right, p_top, p_bottom)
        return F.pad(image, padding)


def get_loader_HAN(args, class_wise=False):
    transform_train = transforms.Compose([
                                    transforms.RandomInvert(p=1.0),
                                    transforms.Grayscale(num_output_channels=1),
                                    transforms.ToTensor(),
                                    SquarePad(),
                                    transforms.Resize((36, 36)),
                                    transforms.RandomCrop((26, 26), pad_if_needed=True),
                                    transforms.Pad((3, 3)),
                                    #  transforms.Resize((34, 34)),
                                    #  transforms.Resize((38, 38)),
                                    #  transforms.RandomCrop(30, padding = 2),
                                    #  transforms.CenterCrop(26),
                                    #  transforms.RandomCrop((26,26)),
                                    transforms.Normalize((0.5,), (0.5,))])
    transform_val = transforms.Compose([
                                    transforms.Resize((args.img_size, args.img_size)),
                                    transforms.RandomInvert(p=1.0),
                                    transforms.Grayscale(num_output_channels=1),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    transform_test = transform_val

    dataset_train = datasets.ImageFolder('../data/HAN/SERI_Test',
                                         transform= transform_train)
    dataset_test = datasets.ImageFolder('../data/HAN/SERI_Test',
                                        transform = transform_test)
    dataset_val = datasets.ImageFolder('../data/HAN/SERI_Test',
                                       transform = transform_val)

    #  dataset_train.targets_np = np.array(dataset_train.targets)
    #  dataset_train = Subset(dataset_train,
                           #  np.where(dataset_train.targets_np < 250)[0])
    
    if class_wise:
        dataset_train.targets_np = np.array(dataset_train.targets)
        loaders = []
        for class_ind in range(10):
            print('class_ind', class_ind)
            dataset_train_subset_ind = Subset(dataset_train, np.where(dataset_train.targets_np == class_ind)[0])
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
