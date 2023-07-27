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


class Qrator(nn.Module):
    def __init__(self,
                 img_size=32,
                 n_qf = 16,
                 levels=2,
                 n_c = 3,
                 n_disc = 0,
                 n_cont = 0,
                 ):
        super().__init__()
        assert img_size in [32, 64, 128]
        self.init_size = img_size // (2**levels)
        self.n_disc = n_disc
        self.n_cont = n_cont
        self.main = nn.Sequential(
                )
        for ind in range(levels):
            self.main.add_module(
                    f'conv_{ind}',
                    nn.Sequential(
                        nn.Conv2d(n_c if ind==0 else n_qf * 2**(ind-1), n_qf * 2**ind, 4, 2, 1),
                        nn.BatchNorm2d(n_qf * 2**ind),
                        #  nn.InstanceNorm2d(n_qf * 2**ind),
                        #  nn.Dropout2d(0.5),
                        nn.LeakyReLU(0.2, inplace=True),
                        )
                    )
        self.main.add_module(
                'final_layer',
                nn.Sequential(
                    Rearrange('b c h w -> b (c h w)'),
                    nn.Linear(n_qf * 2**(levels-1) * self.init_size**2, n_disc + n_cont),
                    )
                )
        initialize_weights(self)

    def forward(self, x):
        output = self.main(x)
        return output

class Generator(nn.Module):
    def __init__(self, 
                img_size=32,
                latent_size = 128, #384, 
                n_gf = 64,
                levels = 2,
                n_c=3,
                n_disc = 0,
                n_cont = 0,
                emb_dim = 100,
                 ):
        super().__init__()
        assert img_size in [32, 64, 128]
        self.n_disc = n_disc
        self.n_cont = n_cont
        self.latent_size = latent_size
        self.init_size = img_size // (2**levels)

        self.emb_c = nn.Sequential(
                nn.Embedding(n_disc, emb_dim),
                Rearrange('b c -> b c 1 1'),
                nn.ConvTranspose2d(emb_dim,
                                   1,
                                   self.init_size, 1, 0),
                #  nn.BatchNorm2d(n_gf * 2**(levels-1)),
                #  nn.LeakyReLU(0.2, inplace=True),
                )
        self.emb_z = nn.Sequential(
                Rearrange('b c -> b c 1 1'),
                nn.ConvTranspose2d(latent_size+n_cont,
                                   n_gf * 2**(levels)-1,
                                   self.init_size, 1, 0),
                #  nn.BatchNorm2d(n_gf * 2**(levels-1)),
                nn.LeakyReLU(0.2, inplace=True),
                )

        self.main = nn.Sequential(
                )
        for layer_ind in range(levels):
            self.main.add_module(
                    f'conv_{layer_ind}',
                    nn.Sequential(
                        nn.ConvTranspose2d(n_gf * 2**(levels-layer_ind),
                                           n_gf * 2**(levels-layer_ind-1) if layer_ind != levels-1 else n_c,
                                           4, 2, 1, bias=False),
                        nn.BatchNorm2d(n_gf * 2**(levels-layer_ind-1)) if layer_ind != levels-1 else nn.Identity(),
                        nn.LeakyReLU(0.2, inplace=True) if layer_ind != levels-1 else nn.Identity(),
                        )
                    )
        self.main.add_module(
                'final_layer',
                nn.Sequential(
                    #  nn.Conv2d(n_gf,
                    #            n_c,
                    #            7, 1, padding='same'),
                    nn.Tanh(),
                    )
                )
        initialize_weights(self)

    def forward(self, x):
        # Split C and Z
        c = x[:, self.latent_size:self.latent_size+self.n_disc]
        x = torch.cat([x[:, :self.latent_size], x[:, self.latent_size+self.n_disc:]], dim=1)
        # embedding
        c = self.emb_c(c.argmax(dim=1))
        x = self.emb_z(x)
        # Merge and compute
        x = torch.cat([x, c], dim=1)
        x = self.main(x)
        return x


#  class Dis_block(nn.Module):
#      def __init__(self, in_channels, out_channels, ksize, normalize=True):
#          super().__init__()
#          self.layers = nn.Sequential(
#                  nn.Conv2d(in_channels, out_channels, ksize, stride=2, padding=1),
#                  )
#          if normalize:
#              #  self.layers.append(nn.BatchNorm2d(out_channels))
#              self.layers.append(nn.InstanceNorm2d(out_channels))
#          self.layers.append(nn.LeakyReLU(0.2))
#
#
#      def forward(self, x):
#          x = self.layers(x)
#          return x
#
#
#
#  def recept_field(ksizes, strides, scale_factor):
#      rf = 1
#      for ksize, stride in zip(ksizes, strides):
#          rf = (rf - 1) * stride + ksize
#      print(f'output_size: {rf/scale_factor}')
#      return rf / scale_factor
#
#  class Discriminator(nn.Module):
#      def __init__(self,
#              scale_factor = 4,
#              levels=3,
#              kernel_size=3,
#              n_df = 64,
#              n_c=3,
#              ):
#          super().__init__()
#
#          self.model = nn.Sequential(
#                  nn.Upsample(scale_factor=scale_factor, mode='bilinear'),
#                  )
#          for layer_ind in range(levels):
#              self.model.add_module(
#                      f'conv_{layer_ind}',
#                      Dis_block(n_c if layer_ind == 0 else n_df * 2**(layer_ind-1),
#                                n_df * 2**layer_ind,
#                                kernel_size,
#                                normalize=layer_ind != 0)
#                      )
#
#          self.model_head = nn.Sequential(
#                  nn.ZeroPad2d((1, 0, 1, 0)),
#                  #  nn.Dropout2d(0.4),
#                  nn.Conv2d( 2**(levels-1) *n_df, 1, 4, padding=1, bias=False),
#                  Reduce('b c h w -> b c', 'mean'),
#                  #  nn.Sigmoid(),
#                  #  Rearrange('b 1 h w -> b (h w)'),
#                  )
#          recept_field( [4] + [kernel_size] *levels, [1] + [2] * levels, scale_factor)
#          initialize_weights(self)
#
#      def forward(self,a,real=True):
#          x = self.model(a)
#          x = self.model_head(x)
#          return x

class Discriminator(nn.Module):
    def __init__(self,
            img_size=32,
            patch_size=6,
            patch_stride=2,
            patch_padding=0,
            levels = 3,
            n_df = 192,
            n_c=3,
            ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride= patch_stride
        self.patch_padding= patch_padding
        self.n_c = n_c


        self.patch_emb = nn.Sequential(
                transforms.Pad(patch_padding, fill=-1),
                nn.Conv2d(n_c, n_df, patch_size, patch_stride, 0),
                nn.LeakyReLU(0.2, inplace=True),
                )
        n = self.patch_emb(torch.zeros(1, n_c, img_size, img_size)).shape[-1]
        self.n_patches = (n)**2
        self.main = nn.Sequential()
        for ind in range(levels -1):
            self.main.add_module(
                    f'conv_{ind}',
                    nn.Sequential(
                        nn.Conv2d(n_df, n_df, 1, 1, 0),
                        nn.InstanceNorm2d(n_df),
                        nn.LeakyReLU(0.2, inplace=True),
                        )
                    )
        self.main_head = nn.Sequential(
                nn.Conv2d(n_df, 1, 1, 1, 0),
                Reduce('b c h w -> b c', 'mean'),
                #  nn.Sigmoid(),
                #  Rearrange('b 1 h w -> b (h w)'),
                )
        initialize_weights(self)
        #  self.pad_real = transforms.Pad(patch_padding, padding_mode='reflect')
        #  self.pad_real = transforms.Pad(patch_padding, fill=0)
        #  self.pad_fake = transforms.Pad(patch_padding, fill=0)

    def forward(self, x, real=True):
        #  x = self.masked_patch(x)
        #  if real:
        #      x = self.pad_real(x)
        #  else:
        #      x = self.pad_fake(x)
        x = self.patch_emb(x) # (batch_size, n_df, n_patches**0.5, n_patches **0.5)
        x = self.main(x) #/ torch.sqrt(torch.tensor(self.n_patches, dtype = torch.float32))
        x = self.main_head(x)
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
            #  return pe.unsqueeze(0)
            return nn.Parameter(pe.unsqueeze(0))

        if learnable == False:
            self.pos_emb2D = _get_sinusoid_encoding_table(d_model, height, width)
        else:
            self.pos_emb2D = nn.Parameter(torch.zeros_like(_get_sinusoid_encoding_table(d_model, height, width)))

    def forward(self, x):
        return torch.cat([x, self.pos_emb2D.expand_as(x)], dim=1)
        #  return x + self.pos_emb2D.to(x.device)
