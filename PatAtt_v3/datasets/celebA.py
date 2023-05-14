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

def get_loader_celeba(args, class_wise=False):
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    dataset_train = datasets.CelebA(
        root='../data',
        split='train',
        target_type='identity',
        transform=transform,
        download=True)
    dataset_test = datasets.CelebA(
        root='../data',
        split='test',
        target_type='identity',
        transform=transform,
        download=True)
    dataset_val = datasets.CelebA(
        root='../data',
        split='valid',
        target_type='identity',
        transform=transform,
        download=True)
    if class_wise:
        loaders = []
        for name, class_ind in dataset_train.class_to_idx.items():
            if name == 'N/A':
                continue
            dataset_train_subset_ind = Subset(dataset_train, np.where(dataset_train.targets == class_ind)[0])
            loader = DataLoader(dataset_train_subset_ind,
                                batch_size=args.test_batch_size,
                                shuffle=True,
                                num_workers=args.num_workers,
                                pin_memory = args.pin_memory)
            loaders.append(loader)
        return loaders, 0, 0
    else:
        train_loader = DataLoader(
            dataset_train,
            sampler=RandomSampler(dataset_train),
            batch_size=args.train_batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=True)
        test_loader = DataLoader(
            dataset_test,
            sampler=SequentialSampler(dataset_test),
            batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False)
        val_loader = DataLoader(
            dataset_val,
            sampler=SequentialSampler(dataset_val),
            batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False)
        return train_loader, val_loader, test_loader




class CelebADataLoader:
    def __init__(self, config):
        self.config = config

        if config.data_mode == "imgs":
            transform = v_transforms.Compose(
                [v_transforms.CenterCrop(64),
                 v_transforms.ToTensor(),
                 v_transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

            dataset = v_datasets.ImageFolder(self.config.data_folder, transform=transform)

            self.dataset_len = len(dataset)

            self.num_iterations = (self.dataset_len + config.batch_size - 1) // config.batch_size

            self.loader = DataLoader(dataset,
                                     batch_size=config.batch_size,
                                     shuffle=True,
                                     num_workers=config.data_loader_workers,
                                     pin_memory=config.pin_memory)
        elif config.data_mode == "numpy":
            raise NotImplementedError("This mode is not implemented YET")
        else:
            raise Exception("Please specify in the json a specified mode in data_mode")

    def plot_samples_per_epoch(self, fake_batch, epoch):
        """
        Plotting the fake batch
        :param fake_batch: Tensor of shape (B,C,H,W)
        :param epoch: the number of current epoch
        :return: img_epoch: which will contain the image of this epoch
        """
        img_epoch = '{}samples_epoch_{:d}.png'.format(self.config.out_dir, epoch)
        v_utils.save_image(fake_batch,
                           img_epoch,
                           nrow=4,
                           padding=2,
                           normalize=True)
        return imageio.imread(img_epoch)

    def make_gif(self, epochs):
        """
        Make a gif from a multiple images of epochs
        :param epochs: num_epochs till now
        :return:
        """
        gen_image_plots = []
        for epoch in range(epochs + 1):
            img_epoch = '{}samples_epoch_{:d}.png'.format(self.config.out_dir, epoch)
            try:
                gen_image_plots.append(imageio.imread(img_epoch))
            except OSError as e:
                pass

        imageio.mimsave(self.config.out_dir + 'animation_epochs_{:d}.gif'.format(epochs), gen_image_plots, fps=2)

    def finalize(self):
        pass
