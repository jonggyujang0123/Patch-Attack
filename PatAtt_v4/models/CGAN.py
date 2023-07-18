"""
mnist tutorial main model
"""
import math 
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from torchvision import transforms
from utils.base_utils import initialize_weights, PixelNorm

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.mode = mode
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

class Qrator(nn.Module):
    def __init__(self,
                 img_size=32,
                 n_qf = 16,
                 levels=2,
                 n_c = 3,
                 n_disc = 0,
                 n_classes = 0,
                 n_cont = 0,
                 ):
        super().__init__()
        assert img_size in [32, 64, 128]
        self.init_size = img_size // (2**levels)
        self.n_disc = n_disc
        self.n_classes = n_classes
        self.n_cont = n_cont
        self.main = nn.Sequential(
                )
        for ind in range(levels):
            self.main.add_module(
                    f'conv_{ind}',
                    nn.Sequential(
                        nn.Conv2d(n_c if ind==0 else n_qf * 2**(ind-1), n_qf * 2**ind, 4, 2, 1),
                        nn.BatchNorm2d(n_qf * 2**ind, 0.8),
                        #  nn.InstanceNorm2d(n_qf * 2**ind),
                        nn.Dropout2d(0.5),
                        nn.LeakyReLU(0.2, inplace=True),
                        )
                    )
        self.main.add_module(
                'final_layer',
                nn.Sequential(
                    Rearrange('b c h w -> b (c h w)'),
                    nn.Linear(n_qf * 2**(levels-1) * self.init_size**2, n_disc * n_classes + n_cont),
                    )
                )
        initialize_weights(self)

    def forward(self, x):
        output = self.main(x)
        return output
        #  return output[:, :self.n_disc*self.n_classes], output[:, self.n_disc*self.n_classes:]

class Generator(nn.Module):
    def __init__(self, 
                img_size=32,
                latent_size = 128, #384, 
                n_gf = 64,
                levels = 2,
                n_c=3,
                n_disc = 0,
                n_cont = 0,
                num_classes = 10,
                ):
        super().__init__()
        assert img_size in [32, 64, 128]
        self.num_classes = num_classes
        self.n_disc = n_disc
        self.n_cont = n_cont
        self.latent_size = latent_size
        self.init_size = img_size // (2**levels)
        self.emb_c = nn.Sequential(
            nn.Embedding(num_classes * n_disc, latent_size+self.n_cont),
            nn.Linear(latent_size+self.n_cont, self.init_size**2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            Rearrange('b (c h w) -> b c h w', c = n_c, h = self.init_size, w = self.init_size),
            )
        self.emb_z = nn.Sequential(
            nn.Linear(latent_size+self.n_cont, self.init_size**2 * (n_gf * 2**levels-1) , bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            Rearrange('b (c h w) -> b c h w', c = n_gf * 2**levels -1, h = self.init_size, w = self.init_size),
            )

        #  self.embed_c = nn.Linear(num_classes * n_disc, latent_size+self.n_cont)
        #  self.embed_z = nn.Linear(latent_size+self.n_cont, latent_size+self.n_cont)
        self.main = nn.Sequential(
                #  Rearrange('b c -> b c 1 1'),
                #  nn.ConvTranspose2d(latent_size + n_cont,
                #                     n_gf * 2**levels,
                #                     self.init_size, 1, 0),
                #  nn.LeakyReLU(0.2, inplace=True),
                )
        for layer_ind in range(levels):
            self.main.add_module(
                    f'conv_{layer_ind}',
                    nn.Sequential(
                        nn.Upsample(scale_factor=2),
                        nn.Conv2d(n_gf * 2**(levels - layer_ind),
                                  n_gf * 2**(levels - layer_ind - 1),
                                  3, 1, 1),
                        nn.BatchNorm2d(n_gf * 2**(levels - layer_ind - 1), 0.8),
                        #  nn.InstanceNorm2d(n_gf * 2**(levels - layer_ind - 1)),
                        #  PixelNorm(),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(n_gf * 2**(levels - layer_ind - 1),
                                  n_gf * 2**(levels - layer_ind - 1),
                                  3, 1, 1),
                        nn.BatchNorm2d(n_gf * 2**(levels - layer_ind - 1), 0.8),
                        #  PixelNorm(),
                        nn.LeakyReLU(0.2, inplace=True),
                        )
                    )
        self.main.add_module(
                'final_layer',
                nn.Sequential(
                    nn.Conv2d(n_gf,
                              n_c,
                              7, 1, padding='same'),
                    nn.Tanh(),
                    )
                )
        initialize_weights(self)

    def forward(self, x):
        c = x[:, self.latent_size:self.latent_size+self.n_disc*self.num_classes]
        x = torch.cat([x[:, :self.latent_size], x[:, self.latent_size+self.n_disc*self.num_classes:]], dim=1)
        c = self.emb_c(c.argmax(dim=1))
        x = self.emb_z(x)
        x = torch.cat([x, c], dim=1)
        #  x = x*c
        x = self.main(x)
        #  c = self.embed_c(c)
        #  x = self.embed_z(x)
        #  x = self.main(x)
        #  x=  self.main(x)
        return x


class Dis_block(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                )
        if normalize:
            #  self.layers.append(nn.BatchNorm2d(out_channels))
            self.layers.append(nn.InstanceNorm2d(out_channels))
        self.layers.append(nn.LeakyReLU(0.2))
    

    def forward(self, x):
        x = self.layers(x)
        return x


class Discriminator(nn.Module):
    def __init__(self,
            img_size=32,
            patch_size=6,
            patch_stride=2,
            patch_padding=0,
            n_df = 64,
            n_c=3,
            ):
        super().__init__()
        self.model = nn.Sequential(
                nn.Upsample(scale_factor=4, mode='bilinear'),
                Dis_block(n_c, n_df, normalize=False),
                Dis_block(n_df, 2*n_df),
                Dis_block(2*n_df, 4*n_df),
                #  Dis_block(4*n_df, 8*n_df),
                nn.ZeroPad2d((1, 0, 1, 0)),
                nn.Conv2d(4*n_df, 1, 3, padding=1, bias=False),
                Reduce('b c h w -> b c', 'mean'),
                #  nn.Sigmoid(),
                #  Rearrange('b 1 h w -> b (h w)'),
                )
        initialize_weights(self)

    def forward(self,a,real=True):
        x = self.model(a)
        return x


#  class Discriminator(nn.Module):
#      def __init__(self,
#              img_size=32,
#              patch_size=6,
#              patch_stride=2,
#              patch_padding=0,
#              n_df = 192,
#              n_c=3,
#              ):
#          super().__init__()
#          self.img_size = img_size
#          self.patch_size = patch_size
#          self.patch_stride= patch_stride
#          self.patch_padding= patch_padding
#          self.n_c = n_c
#
#          #  self.n_patches = (n)**2
#          n_patches = (img_size + 2 * patch_padding - patch_size) // patch_stride + 1
#
#          self.patch_emb = nn.Sequential(
#                  #  nn.Conv2d(n_c, n_df, patch_size, patch_stride, patch_padding),
#                  nn.Conv2d(n_c, n_df, patch_size, patch_stride, 0),
#          n = self.patch_emb(torch.zeros(1, n_c, img_size, img_size)).shape[-1]
#          self.main = nn.Sequential(
#                  #  PositionalEncoding2D(n_df, n, n, learnable=False),
#                  nn.Conv2d(n_df, n_df, 1, 1, 0),
#                  nn.BatchNorm2d(n_df),
#                  nn.LeakyReLU(0.2, inplace=True),
#                  #  PixelNorm(),
#                  #  nn.Dropout2d(0.25),
#                  nn.Conv2d(n_df, n_df, 1, 1, 0),
#                  nn.BatchNorm2d(n_df),
#                  nn.LeakyReLU(0.2, inplace=True),
#                  #  PixelNorm(),
#                  #  nn.Dropout2d(0.25),
#                  nn.Conv2d(n_df, n_df, 1, 1, 0),
#                  nn.BatchNorm2d(n_df),
#                  nn.LeakyReLU(0.2, inplace=True),
#                  #  PixelNorm(),
#                  #  nn.Dropout2d(0.25),
#                  nn.Conv2d(n_df, 1, 1, 1, 0),
#                  #  nn.Dropout2d(0.5),
#                  Reduce('b c h w -> b c', 'mean'),
#                  nn.Sigmoid(),
#  #                Rearrange('b d ph pw -> (b ph pw) d'),
#                  )
#          initialize_weights(self)
#          #  self.pad_real = transforms.Pad(patch_padding, padding_mode='reflect')
#          self.pad_real = transforms.Pad(patch_padding, fill=0)
#          self.pad_fake = transforms.Pad(patch_padding, fill=0)
#
#      def forward(self, x, real=True):
#          #  x = self.masked_patch(x)
#          if real:
#              x = self.pad_real(x)
#          else:
#              x = self.pad_fake(x)
#          x = self.patch_emb(x) # (batch_size, n_df, n_patches**0.5, n_patches **0.5)
#          x = self.main(x) #/ torch.sqrt(torch.tensor(self.n_patches, dtype = torch.float32))
#          return x

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
            #  return pe.unsqueeze(0)
            return nn.Parameter(pe.unsqueeze(0))

        if learnable == False:
            self.pos_emb2D = _get_sinusoid_encoding_table(d_model, height, width)
        else:
            self.pos_emb2D = nn.Parameter(torch.zeros_like(_get_sinusoid_encoding_table(d_model, height, width)))

    def forward(self, x):
        return torch.cat([x, self.pos_emb2D.expand_as(x)], dim=1)
        #  return x + self.pos_emb2D.to(x.device)
