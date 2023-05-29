"""
mnist tutorial main model
"""
import math 
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from torchvision import transforms
import numpy as np


class Cutout(nn.Module):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, 
                 n_holes = 4, 
                 length = 0.4):
        super(Cutout, self).__init__()
        self.max_holes = n_holes
        self.max_length = length

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
        length_h = int(img.size(2) * torch.rand(1) * self.max_length)
        length_w = int(img.size(3) * torch.rand(1) * self.max_length)

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
        img = img * mask 

        return img

    def __repr__(self):
        return self.__class__.__name__ + '(n_holes={0}, length={1})'.format(self.n_holes, self.length)



def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class classifier_transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.fake_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=.5),
                transforms.RandomVerticalFlip(p=.5),
                transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
                ])
    def forward(self,x):
        return self.fake_transform(x)


class transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.fake_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=.5),
                transforms.RandomVerticalFlip(p=.5),
                transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
                ])
    def forward(self,x):
        return self.fake_transform(x)

class Generator(nn.Module):
    def __init__(self, 
                img_size=32,
                num_classes = 10,
                latent_size = 128, #384, 
                n_gf = 64,
                levels = 2,
                n_c=3,
                ):
        super().__init__()
        assert img_size in [32, 64, 128]
        self.num_classes = num_classes
        self.init_size = img_size // (2**levels) 
        self.embed = nn.Embedding(num_classes, latent_size)
        self.main = nn.Sequential(
                Rearrange('b c -> b c 1 1'),
                nn.ConvTranspose2d(latent_size + num_classes, n_gf * 2**levels, self.init_size, 1, 0),
                nn.BatchNorm2d(n_gf * 2**levels),
                nn.LeakyReLU(0.2, inplace=True),
                #  nn.ReLU(inplace=True),
                #  PixelNorm(),
                #  nn.Linear(latent_size, 1024),
                #  nn.BatchNorm1d(1024),
                #  nn.ReLU(inplace=True),
                #  #  nn.LeakyReLU(0.2, inplace=True),
                #  nn.Linear(1024, 2 ** (levels-1) * n_gf * self.init_size**2),
                #  nn.BatchNorm1d(2 ** (levels-1) * n_gf * self.init_size**2),
                #  nn.ReLU(inplace=True),
                #  nn.LeakyReLU(0.2, inplace=True),
                #  Rearrange('b (c h w) -> b c h w', h=self.init_size, w=self.init_size),
                )
        for layer_ind in range(levels):
            self.main.add_module(
                    f'conv_{layer_ind}',
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            n_gf * 2**(levels - layer_ind),
                            n_gf * 2 **(levels - layer_ind -1),
                            4, 2, 1),
                        nn.BatchNorm2d(n_gf * 2 **(levels - layer_ind -1)),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(n_gf * 2**(levels - layer_ind - 1),
                                  n_gf * 2**(levels - layer_ind - 1) if layer_ind < levels - 1 else n_c,
                                  3, 1, 1),
                        nn.BatchNorm2d(n_gf * 2**(levels - layer_ind - 1)) if layer_ind < levels - 1 else nn.Identity(),
                        nn.LeakyReLU(0.2, inplace=True) if layer_ind < levels - 1 else nn.Identity(),
                        nn.Tanh() if layer_ind == levels - 1 else nn.Identity(),
                        #  nn.ReLU(inplace=True),
                        #  PixelNorm(),
                        #  nn.Conv2d(n_gf * 2**(levels - layer_ind - 1),
                                  #  n_gf * 2**(levels - layer_ind - 1),
                                  #  3, 1, 1),
                        #  nn.BatchNorm2d(n_gf * 2**(levels - layer_ind - 1)),
                        #  nn.ReLU(inplace=True),
                        #  nn.LeakyReLU(0.2, inplace=True),
                        )
                    )
            #  self.main.append(PixelNorm())
            #  if layer_ind == levels - 1:
            #      self.main.append(nn.Tanh())
            #  else:
            #      #  self.main.append(nn.BatchNorm2d(n_gf * 2**(levels - layer_ind - 1)))
            #      self.main.append(nn.LeakyReLU(0.2, inplace=True))
            #      self.main.append(PixelNorm())
            #      #  self.main.append(nn.ReLU(inplace=True))
        initialize_weights(self)

    def forward(self, x, c):
        #  c = self.embed(c)
        #  x = self.main(x*c)
        c = F.one_hot(c, num_classes = self.num_classes).float()
        x = self.main(torch.cat([x, c], dim=1))
        return x



class Discriminator(nn.Module):
    def __init__(self,
            img_size=32,
            patch_size=6,
            patch_stride=2,
            patch_padding=2,
            n_df = 192,
            n_c=3,
            ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride= patch_stride
        self.transform = transform()
        self.patch_emb = nn.Sequential(
                #  transform(),
                nn.Conv2d(n_c, n_df, patch_size, patch_stride, patch_padding),
                nn.LeakyReLU(0.2, inplace=True),
                #  nn.BatchNorm2d(n_df),
                #  nn.Conv2d(n_c, n_df, patch_size, patch_stride, patch_padding, padding_mode='replicate'),
                #  nn.LeakyReLU(0.2, inplace=True),
                #  nn.BatchNorm2d(n_df),
                )
        n = self.patch_emb(torch.zeros(1, n_c, img_size, img_size)).shape[-1]
        self.n_patches = (n)**2
#        padding_size = (n * patch_stride - (img_size - (patch_size- patch_stride)))//2
#        self.pos_emb1D = nn.Parameter(torch.randn(1, n_df, n, n)* scale)

        self.main = nn.Sequential(
                #  PositionalEncoding2D(n_df, n, n, learnable=False),
                nn.Conv2d(n_df, n_df, 1, 1, 0),
                nn.BatchNorm2d(n_df),
                nn.LeakyReLU(0.2, inplace=True),
                #  nn.Conv2d(n_df, n_df, 1, 1, 0),
                #  nn.BatchNorm2d(n_df),
                #  nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(n_df, n_df, 1, 1, 0),
                nn.BatchNorm2d(n_df),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(n_df, n_df, 1, 1, 0),
                nn.BatchNorm2d(n_df),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(n_df, 1, 1, 1, 0, bias=False),
                Reduce('b c h w -> b c', 'sum'),
#                nn.Dropout(0.5),
                nn.Sigmoid(),
#                Rearrange('b d ph pw -> (b ph pw) d'),
                )
        initialize_weights(self)

    def forward(self, x, transform = True):
        if transform:
            x = self.transform(x)
        x = self.patch_emb(x) # (batch_size, n_df, n_patches**0.5, n_patches **0.5)
        x = self.main(x) #/ torch.sqrt(torch.tensor(self.n_patches, dtype = torch.float32))
        return x

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height, width, dropout=0.1, learnable=False):
        super(PositionalEncoding2D, self).__init__()

        def _get_sinusoid_encoding_table(d_model, height, width):
            if d_model % 4 != 0:
                raise ValueError("Cannot use sin/cos positional encoding with "
                                 "odd dimension (got dim={:d})".format(d_model))
            pe = torch.zeros(d_model, height, width)
            # Each dimension use half of d_model
            d_model = int(d_model / 2)
            max_len = height * width
            div_term = torch.exp(torch.arange(0., d_model, 2) *
                                 -(math.log(max_len) / d_model))
            pos_w = torch.arange(0., width).unsqueeze(1)
            pos_h = torch.arange(0., height).unsqueeze(1)
            pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
            pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
            pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
            pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
            return pe.unsqueeze(0)

        if learnable == False:
            self.pos_emb2D = _get_sinusoid_encoding_table(d_model, height, width)
        else:
#            self.pos_emb2D = nn.Parameter(_get_sinusoid_encoding_table(d_model, height, width))
            self.pos_emb2D = nn.Parameter(torch.zeros_like(_get_sinusoid_encoding_table(d_model, height, width)))

    def forward(self, x):
        #  return x + self.pos_emb2D.to(x.device)
        #  return x * self.pos_emb2D.to(x.device)
        return torch.cat([x, self.pos_emb2D.to(x.device).expand_as(x)], dim=1)
        #  return torch.cat([x, self.pos_emb2D.to(x.device).tile([x.shape[0], 1, 1, 1])], dim=1)
#        return self.dropout(x * self.pos_emb2D.to(x.device))



