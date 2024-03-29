import torch
import numpy as np
import random
from torch.optim.lr_scheduler import LambdaLR
import math
import os 
import shutil
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics import StructuralSimilarityIndexMeasure
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

def sample_noise(n_disc, n_z, batch_size, device):
    z = torch.randn(batch_size, n_z, device=device)
    idx = torch.randint(n_disc, size = (batch_size,)).to(device)
    disc_c = F.one_hot(idx, n_disc).float().to(device).view(batch_size, -1)
    noise = torch.cat([z, disc_c], dim=1)
    return noise, idx

class AddGaussianNoise(nn.Module):
    def __init__(self, mean=0., std=1.):
        super(AddGaussianNoise, self).__init__()
        self.std = std
        self.mean = mean

    def forward(self, x):
        noise = torch.randn(x.size(), device=x.device) * self.std + self.mean
        return x + noise



class Sharpness(nn.Module):
    def __init__(self,
                 min_s = 0.5,
                 max_s = 2.0,
                 ):
        super(Sharpness, self).__init__()
        self.min_s = min_s
        self.max_s = max_s

    def forward(self, x):
        s = torch.rand(1) * (self.max_s - self.min_s) + self.min_s
        x = transforms.functional.adjust_sharpness(x, s)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(min_s={}, max_s={})'.format(self.min_s, self.max_s)



class Cutout(nn.Module):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self,
                 n_holes = 4,
                 length = 0.25,
                 fill = 0.0,):
        super(Cutout, self).__init__()
        self.n_holes = n_holes
        self.length = length
        self.max_holes = n_holes + 1
        self.max_length = length
        self.fill = fill

    def forward(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        n_holes = torch.randint(0, self.max_holes, [])
        h = img.size(2)
        w = img.size(3)
        length_h = int(h * torch.rand(1) * self.max_length)
        length_w = int(w* torch.rand(1) * self.max_length)

        mask = torch.ones((h, w), dtype=torch.float32)

        for n in range(n_holes):
            y = torch.randint(0,h,())
            x = torch.randint(0,w,())

            y1 = torch.clip(y - length_h // 2, 0, h)
            y2 = torch.clip(y + length_h // 2, 0, h)
            x1 = torch.clip(x - length_w // 2, 0, w)
            x2 = torch.clip(x + length_w // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.
        mask = mask.expand_as(img).to(img.device)
        if self.fill == 'random':
            img = img * mask + torch.rand_like(img) * (1. - mask)
        else:
            img = img * mask + self.fill * (1. - mask)

        return img

    def __repr__(self):
        return self.__class__.__name__ + '(n_holes={0}, length={1})'.format(self.n_holes, self.length)

class Mixup(nn.Module):
    def __init__(
            self,
            alpha=1.0,
            ):
        super(Mixup, self).__init__()
        self.alpha = alpha
    
    def forward(self, x, y):
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def __repr__(self):
        return self.__class__.__name__ + '(alpha={})'.format(self.alpha)

class CutMix(nn.Module):
    def __init__(
            self,
            alpha=1.0,
            ):
        super(CutMix, self).__init__()
        self.alpha = alpha

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2

    def forward(self, x, y):
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
        mask = torch.zeros([1,1, x.size()[-2], x.size()[-1]]).to(x.device).float()
        mask[:, :, bbx1:bbx2, bby1:bby2] = 1
        x = x * (1-mask) + x[index, ...] * mask
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        return x, y, y[index], lam

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            #  if m.bias is not None:
            #      m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            #  if m.bias is not None:
            #      m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            #  if m.bias is not None:
            #      m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            #  if m.bias is not None:
            #      m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.normal_(1.0, 0.02)
            #  if m.bias is not None:
            #      m.bias.data.zero_()

def noisy_labels(y, p_flip):
    # choose labels to flip
    flip_ix = np.random.choice(y.size(0), int(y.size(0) *  p_flip))
    # invert the labels in place
    y[flip_ix] = 1 - y[flip_ix]
    return y

def get_fid_score():
    fid = FrechetInceptionDistance(feature=64)
    # generate two slightly overlapping image intensity distributions
    imgs_dist1 = torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)
    imgs_dist2 = torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8)
    fid.update(imgs_dist1, real=True)
    fid.update(imgs_dist2, real=False)
    xx = fid.compute()
    return xx


def get_ssim():
    preds = torch.rand([1, 3, 256, 256]).tile(100,1,1,1)
    target = preds * 0.75
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    xx = ssim(preds, target)
    return xx 



def get_data_loader(args, dataset_name = None, class_wise=False):
    dataset_name = args.dataset if dataset_name is None else dataset_name
    if dataset_name in ['cifar10']:
        from datasets.cifar10 import get_loader_cifar10 as get_loader
    if dataset_name in ['cifar100']:
        from datasets.cifar100 import get_loader_cifar100 as get_loader
    if dataset_name in ['mnist']:
        from datasets.mnist import get_loader_mnist as get_loader
    if dataset_name in ['emnist']:
        from datasets.emnist import get_loader_emnist as get_loader
    if dataset_name in ['fashion']:
        from datasets.fashion_mnist import get_loader_fashion_mnist as get_loader
    if dataset_name in ['kmnist']:
        from datasets.kmnist import get_loader_kmnist as get_loader
    if dataset_name == 'celeba':
        from datasets.celebA import get_loader_celeba as get_loader
    if dataset_name == 'LFW':
        from datasets.LFW import get_loader_LFW as get_loader
    if dataset_name == 'HAN':
        from datasets.HAN import get_loader_HAN as get_loader
    if dataset_name == 'caltech101':
        from datasets.caltech101 import get_loader_caltech101 as get_loader
    if dataset_name == 'caltech256':
        from datasets.caltech256 import get_loader_caltech256 as get_loader
    return get_loader(args, class_wise= class_wise)


def load_ckpt(checkpoint_fpath, is_best =False):
    """
    Latest checkpoint loader
        checkpoint_fpath : 
    :return: dict
        checkpoint{
            model,
            optimizer,
            epoch,
            scheduler}
    example :
    """
    if is_best:
        ckpt_path = checkpoint_fpath+'/'+'best.pt'
    else:
        ckpt_path = checkpoint_fpath+'/'+'checkpoint.pt'
    try:
        print(f"Loading checkpoint '{ckpt_path}'")
        checkpoint = torch.load(ckpt_path)
    except:
        print(f"No checkpoint exists from '{ckpt_path}'. Skipping...")
        print("**First time to train**")
    return checkpoint


def save_ckpt(checkpoint_fpath, checkpoint, is_best=False):
    """
    Checkpoint saver
    :checkpoint_fpath : directory of the saved file
    :checkpoint : checkpoiint directory
    :return:
    """
    ckpt_path = checkpoint_fpath+'/'+'checkpoint.pt'
    # Save the state
    if not os.path.exists(checkpoint_fpath):
        os.makedirs(checkpoint_fpath)
    torch.save(checkpoint, ckpt_path)
    # If it is the best copy it to another file 'model_best.pth.tar'
#    print("Checkpoint saved successfully to '{}' at (epoch {})\n"
#        .format(ckpt_path, checkpoint['epoch']))
    if is_best:
        ckpt_path_best = checkpoint_fpath+'/'+'best.pt'
#        print("This is the best model\n")
        shutil.copyfile(ckpt_path,
                        ckpt_path_best)


def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

class AverageMeter:
    """
    Class to be an average meter for any average metric like loss, accuracy, etc..
    """

    def __init__(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def val(self):
        return self.avg

def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.

    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch

    :param output: output is the prediction of the model e.g. scores, logits, raw y_pred before normalization or getting classes
    :param target: target is the truth
    :param topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
    e.g. in top 2 it means you get a +1 if your models's top 2 predictions are in the right label.
    So if your model predicts cat, dog (0, 1) and the true label was bird (3) you get zero
    but if it were either cat or dog you'd accumulate +1 for that example.
    :return: list of topk accuracy [top1st, top2nd, ...] depending on your topk input
    """
    with torch.no_grad():
        # ---- get the topk most likely labels according to your model
        # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
        maxk = max(topk)  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)

        # get top maxk indicies that correspond to the most likely probability scores
        # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

        # - get the credit for each example if the models predictions is in maxk values (main crux of code)
        # for any example, the model will get credit if it's prediction matches the ground truth
        # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
        # if the k'th top answer of the model matches the truth we get 1.
        # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
        target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
            # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
            topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc)
        return list_topk_accs  # list of topk accuracies for entire batch [topk1, topk2, ... etc]

def get_accuracy(output, label):
    pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
    acc = pred.eq(label.view_as(pred)).sum().item()/pred.size(0)
    return acc

class ConstantLRSchedule(LambdaLR):
    """ Constant learning rate schedule.
    """
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLRSchedule, self).__init__(optimizer, lambda _: 1.0, last_epoch=last_epoch)


class WarmupConstantSchedule(LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupConstantSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        return 1.


class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


def set_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad= flag
