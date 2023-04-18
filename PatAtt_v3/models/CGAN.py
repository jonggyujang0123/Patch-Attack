"""
mnist tutorial main model
"""
import math 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from torchvision.transforms import Pad 
from einops.layers.torch import Rearrange


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
#            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
#            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
#            m.bias.data.zero_()

class Generator(nn.Module):
    def __init__(self, 
                img_size=32,
                n_classes = 10,
                latent_size = 128, #384, 
                n_gf = 64,
                levels = 2,
                n_c=3,
                len_code = 10
                ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.n_classes = n_classes
        self.init_size = img_size // (2**levels) 
        assert img_size in [32, 64, 128]
        #levels = int(math.log2(img_size)) - 2# if image size is 32, number of levels is 3 {0, 1, 2} with channel coefficients of  {1, 2, 4}
        self.label_layer = nn.Embedding(n_classes, n_classes)
        self.latent_layer = nn.Sequential(
                nn.Linear(latent_size + n_classes + len_code, n_gf * 2**(levels) * self.init_size**2, bias=False),
                nn.BatchNorm1d(n_gf * 2**(levels) * self.init_size**2),
#                nn.LeakyReLU(0.2, inplace=True),
                nn.ReLU(inplace=True),
                nn.Linear(n_gf * 2**(levels) * self.init_size**2, n_gf * 2**(levels) * self.init_size**2, bias=False),
                nn.BatchNorm1d(n_gf * 2**(levels) * self.init_size**2),
#                nn.LeakyReLU(0.2, inplace=True),
                nn.ReLU(inplace=True),
                Rearrange('b (c h w) -> b c h w', h=self.init_size, w=self.init_size),
#                nn.ConvTranspose2d(latent_size, n_gf * 2**(levels), self.init_size, 1, 0),
#                nn.BatchNorm2d(n_gf * 2**(levels)),
#                nn.LeakyReLU(0.2, inplace=True),
                )
        self.main = nn.Sequential()
        for layer_ind in range(levels):
            self.main.append(
                    nn.ConvTranspose2d(
                        n_gf * 2**(levels - layer_ind ),
                        n_gf * 2 **(levels - layer_ind -1 ) if layer_ind < levels - 1 else n_c,
                        4, 2, 1, bias=False)
                    )
            if layer_ind == levels - 1:
                self.main.append(nn.Tanh())
            else:
                self.main.append(nn.BatchNorm2d(n_gf * 2**(levels - 1 - layer_ind)))
#                self.main.append(nn.LeakyReLU(0.2, inplace=True))
                self.main.append(nn.ReLU(inplace=True))

        initialize_weights(self)

    def forward(self, x, c, y_cont):
        c = F.one_hot(c, num_classes = self.n_classes).float() 
        x = self.latent_layer(torch.cat([x, c, y_cont], dim=1))
#        c = self.label_layer(c)
#        x = self.latent_layer(torch.cat([x*c, y_cont], dim=1))
        return self.main(x)

#class Generator(nn.Module):
#    def __init__(self, 
#                img_size=32,
#                n_classes = 10,
#                latent_size = 128, #384, 
#                n_gf = 64,
#                levels = 2,
#                n_c=3
#                ):
#        super().__init__()
#        self.layers = nn.ModuleList()
#        self.init_size = img_size // (2**levels) 
#        self.n_classes = n_classes
#        assert img_size in [32, 64, 128]
#        #levels = int(math.log2(img_size)) - 2# if image size is 32, number of levels is 3 {0, 1, 2} with channel coefficients of  {1, 2, 4}
#        
#        self.label_layer = nn.Embedding(n_classes, latent_size)
#        self.latent_layer = nn.Sequential(
#                nn.ConvTranspose2d(latent_size, n_gf * 2**(levels), self.init_size, 1, 0),
#                nn.BatchNorm2d(n_gf * 2**(levels)),
#                )
#                nn.LeakyReLU(0.2, inplace=True),
#        self.main = nn.Sequential()
#        self.main.append(
#                nn.BatchNorm2d(n_gf * 2**(levels))
#                )
#        for layer_ind in range(levels):
#            block = nn.Sequential(
#                    nn.Upsample(scale_factor=2),
#                    nn.Conv2d(
#                        n_gf * 2 **(levels - layer_ind), n_gf * 2 **(levels - 1 - layer_ind), 3, stride=1, padding=1),
#                    nn.BatchNorm2d(n_gf* 2**(levels-1-layer_ind), 0.8),
#                    nn.LeakyReLU(0.2,inplace=True)
#                    )
#            self.main.append(block)
##            self.main.add_module(name=f'block{layer_ind}', module=block)
#
#        self.main.append(
#                nn.Sequential(
#                    nn.Conv2d(n_gf, n_c, 3, stride=1, padding=1),
#                    nn.Tanh()
#                    )
#                )
#        initialize_weights(self)
#
#    def forward(self, x, c):
##        c = self.label_layer(F.one_hot(c, num_classes = self.n_classes).float().unsqueeze(2).unsqueeze(3)) 
#        c = self.label_layer(c) 
#        x = self.latent_layer((x*c).unsqueeze(2).unsqueeze(3))
#        return self.main(x)

class Qrator(nn.Module):
    def __init__(
            self,
            img_size=32,
            n_qf = 16,
            levels=3,
            n_c = 1,
            len_code = 2
            ):
        super().__init__()
        self.img_size = img_size
        self.bs = math.ceil(img_size / (2 ** levels) )

        def discriminator_block(in_filters, out_filters, bn = True):
            block = nn.Sequential(nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                                  )
            if bn:
                block.append(nn.BatchNorm2d(out_filters,0.8))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block

        self.main = nn.Sequential(
                discriminator_block(n_c, n_qf, bn=False))
        for layer_ind in range(1,levels):
            self.main.append(
                    discriminator_block(n_qf * 2 **(layer_ind - 1), n_qf * 2 ** (layer_ind), bn = True if layer_ind < levels - 1 else False)
                    )
        self.main.append(
                nn.Sequential(
                    nn.Flatten(start_dim = 1, end_dim=-1),
                    nn.Linear(self.bs ** 2 * n_qf * 2 ** (levels-1), len_code),
                    )
                )
        initialize_weights(self)

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self,
            img_size=32,
            patch_size=6,
            patch_stride=2,
            n_df = 192,
            n_c=3,
            ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride= patch_stride
        # Compute Padding size for given img_size, patch_size, patch_stride
        # n * patch_stride + (patch_size-patch_stride)  = 2*padding_size + img_size: what is the smallest non-negative integer p?
        n = math.ceil((img_size - (patch_size - patch_stride)) / patch_stride)
        self.n_patches = (n)**2
        padding_size = (n * patch_stride - (img_size - (patch_size- patch_stride)))//2

        self.patch_emb = nn.Sequential(
#                nn.Conv2d(n_c, n_df, patch_size, patch_stride, padding_size),
                nn.utils.spectral_norm(nn.Conv2d(n_c, n_df, patch_size, patch_stride, padding_size, bias=False)),
#                nn.BatchNorm2d(n_df),
#                nn.LeakyReLU(0.2, inplace=True),
                PositionalEncoding2D(n_df, n, n, learnable=False),
                )
#        self.pos_emb1D = nn.Parameter(torch.randn(1, n_df, n, n)* scale)
        self.main = nn.Sequential(
                Rearrange('b d ph pw -> (b ph pw) d'),
                nn.Linear(2*n_df, n_df),
                nn.BatchNorm1d(n_df),
                nn.LeakyReLU(0.1, inplace=True),
#                nn.Dropout(0.5),
                nn.Linear(n_df, n_df),
                nn.BatchNorm1d(n_df),
                nn.LeakyReLU(0.1, inplace=True),
#                nn.Dropout(0.4),
                nn.Linear(n_df, n_df),
                nn.BatchNorm1d(n_df),
                nn.LeakyReLU(0.1, inplace=True),
#                nn.Dropout(0.4),
                nn.Linear(n_df, 1),
                nn.Sigmoid()
                )
        initialize_weights(self)

    def forward(self, x):
        x = self.patch_emb(x) # (batch_size, n_df, n_patches**0.5, n_patches **0.5)
        return self.main(x)

#class Discriminator(nn.Module):
#    def __init__(self,
#            img_size=32,
#            patch_size=6,
#            patch_stride=2,
#            levels=4,
#            n_df = 16,
#            n_c=3,
#            batch_size = 64,
#            keep_ratio = 0.03):
#        super().__init__()
#        self.pad = Pad(patch_size - patch_stride, fill=0)
#        self.img_size = img_size
#        self.patch_size = patch_size
#        self.patch_stride= patch_stride
#        self.bs = math.ceil(patch_size / (2 ** levels) )
#        self.n_patches = ((img_size + 2* (patch_size - patch_stride) -patch_size) // patch_stride + 1) **2
#        self.keep_ratio = keep_ratio
#        pos_emb = positionalencoding2d(n_df * (patch_size//2) **2 , int(self.n_patches**0.5), int(self.n_patches**0.5) ).permute(1, 2, 0).reshape([1, self.n_patches, n_df, patch_size//2, patch_size//2]).tile([batch_size, 1, 1, 1, 1])
#
##        pos_emb = nn.Parameter(torch.randn( 1, self.n_patches, n_df, 1, 1 )).tile([batch_size, 1, 1, 1, 1])
#        self.pos_emb1D = pos_emb.reshape(pos_emb.shape[0] * pos_emb.shape[1], pos_emb.shape[2], patch_size//2, patch_size//2)
#
#        def discriminator_block(in_filters, out_filters, bn = True):
#            block = nn.Sequential(nn.Conv2d(in_filters, out_filters, 3, 2, 1),
#                    nn.LeakyReLU(0.2, inplace=True),
#                    nn.Dropout2d(0.25))
#            if bn:
#                block.append(nn.BatchNorm2d(out_filters,0.8))
#            return block
#
#        self.main_1 = nn.Sequential(
#                discriminator_block(n_c, n_df, bn=False))
#        self.main_2 = nn.Sequential()
#        for layer_ind in range(1,levels):
#            self.main_2.append(
#                    discriminator_block(n_df * 2 **(layer_ind - 1), n_df * 2 ** (layer_ind), bn = True if layer_ind < levels - 1 else False)
#                    )
#        self.main_2.append(
#                nn.Sequential(
#                    nn.Flatten(start_dim = 1, end_dim=-1),
#                    nn.Linear(self.bs ** 2 * n_df * 2 ** (levels-1), 1),
#                    nn.Sigmoid()
#                    )
#                )
#
#
#        self.weight_init()
#
#    def forward(self, x):
#        x = self.pad(x).unfold(2, self.patch_size, self.patch_stride).unfold(3, self.patch_size, self.patch_stride)
##        print(x.shape)
#        x = x.contiguous().view(
#                x.shape[0], x.shape[2] * x.shape[3], x.shape[1], x.shape[4], x.shape[5]
#                ) # [b, n_pat, c, p_size, p_size]
#        x = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])
#        # index 
##        print(self.n_patches)
#
##        rand_ind = np.random.choice(np.arange(x.shape[0]), int(x.shape[0]* self.keep_ratio))
##        x = x[rand_ind, :, :, :]
##        pos_emb = self.pos_emb1D[rand_ind, :, :, :].to(x.device)
#        x = self.main_1(x)
#        x = x + self.pos_emb1D.to(x.device)
##        return self.last(self.main(x))
#        return self.main_2(x)
##        return self.last(self.main(x))
#
#    def weight_init(self):
#        for m in self._modules:
#            normal_init(self._modules[m])


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
            self.pos_emb2D = nn.Parameter(_get_sinusoid_encoding_table(d_model, height, width))
        self.dropout = nn.Dropout(p=dropout) 

    def forward(self, x):
#        return self.dropout(x + self.pos_emb2D.to(x.device))
        return torch.cat([x, self.pos_emb2D.to(x.device).tile([x.shape[0], 1, 1, 1])], dim=1)
#        return self.dropout(x * self.pos_emb2D.to(x.device))
