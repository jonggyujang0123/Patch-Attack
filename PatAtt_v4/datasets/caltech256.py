"""
caltech101 dataset
"""
import logging
import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torch.utils.data import Subset
logger = logging.getLogger()


import os
import os.path
from typing import Any, Callable, List, Optional, Tuple, Union

from PIL import Image

from torchvision.datasets.utils import download_url, check_integrity, verify_str_arg, download_and_extract_archive
from torchvision.datasets.vision import VisionDataset

class Caltech256(VisionDataset):
    """`Caltech 256 <https://data.caltech.edu/records/20087>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``caltech256`` exists or will be saved to if download is set to True.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(os.path.join(root, "caltech256"), transform=transform, target_transform=target_transform)
        os.makedirs(self.root, exist_ok=True)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        self.categories = sorted(os.listdir(os.path.join(self.root, "256_ObjectCategories")))
        self.index: List[int] = []
        self.y = []
        for (i, c) in enumerate(self.categories):
            n = len(
                [
                    item
                    for item in os.listdir(os.path.join(self.root, "256_ObjectCategories", c))
                    if item.endswith(".jpg")
                ]
            )
            self.index.extend(range(1, n + 1))
            self.y.extend(n * [i])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = Image.open(
            os.path.join(
                self.root,
                "256_ObjectCategories",
                self.categories[self.y[index]],
                f"{self.y[index] + 1:03d}_{self.index[index]:04d}.jpg",
            )
        )

        target = self.y[index]

        if self.transform is not None:
            if img.mode == "L":
                img = self.transform(img.convert("RGB"))
            else:
                img = self.transform(img)
        #  if self.transform is not None:
        #      img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def _check_integrity(self) -> bool:
        # can be more robust and check hash of files
        return os.path.exists(os.path.join(self.root, "256_ObjectCategories"))

    def __len__(self) -> int:
        return len(self.index)

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_and_extract_archive(
            "https://drive.google.com/file/d/1r6o0pSROcV1_VwT4oSjA2FBUSCWGuxLK",
            self.root,
            filename="256_ObjectCategories.tar",
            md5="67b4f42ca05d46448c6bb8ecd2220f6d",
        )



def get_loader_caltech256(args, class_wise=False):
    transform_train = transforms.Compose([
        #transforms.RandomResizedCrop((args.img_size, args.img_size), scale= (0.05, 1.0)),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.CenterCrop(128),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])
    


    dataset_train = Caltech256(root="../data",
                                        download=True,
                                        transform = transform_train,
                                        ) 
    if class_wise:
        loaders = []
        for class_ind in range(256):
            dataset_train_subset_ind = Subset(dataset_train, np.where(np.array(dataset_train.y) == class_ind)[0])
            loader = DataLoader(dataset_train_subset_ind,
                                batch_size=args.test_batch_size,
                                shuffle=True,
                                num_workers=args.num_workers,
                                pin_memory = args.pin_memory)
            loaders.append(loader)
        return loaders, 0, 0
    else:
        cls_ignore = [55, 59, 104, 250, 251, 128, 249, 206, 196, 157, 151, 145, 144, 101, 79, 71, 223 ]
        train_inds = np.where(np.isin(np.array(dataset_train.y), cls_ignore, invert=True))[0]
        dataset_train = Subset(dataset_train, train_inds)
        train_loader = DataLoader(dataset_train,
                                  sampler=RandomSampler(dataset_train),
                                  batch_size = args.train_batch_size,
                                  num_workers = args.num_workers,
                                  pin_memory=args.pin_memory)
        return train_loader, 0, 0
