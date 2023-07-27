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

class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        self.n_holes = np.random.randint(0, 4)
        self.length = np.random.randint(6, 16)
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

def get_loader_celeba(args, class_wise=False):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(1.0,1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        #  transforms.RandomHorizontalFlip(p=0.5),
        #  transforms.CenterCrop(128),
        #  transforms.Resize(64),
        transforms.ToTensor(),
        #  Cutout(n_holes=3, length=16),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    transform_test = transforms.Compose([
        #  transforms.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(1.0,1.0)),
        #  transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        #  transforms.CenterCrop(128),
        transforms.Resize(224),
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
        split='train',
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
        for class_ind in range(1000):
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
        n_target = int(n_total * 0.9)
        #  n_val = n_total - n_target
        mask = torch.zeros(n_total).bool()
        mask_ind = torch.randperm(n_total)[:n_target]
        mask[mask_ind] = True
        dataset_train = Subset(dataset_train, 
                               np.where(mask * (dataset_train.identity[:,0] < 1000))[0])
        dataset_test = Subset(dataset_test,
                              np.where( (~mask) * (dataset_test.identity[:,0] < 1000))[0])
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
