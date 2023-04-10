"""
mnist tutorial main model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
class Generator(nn.Module):
    def __init__(self, 
                patch_size=4,
                patch_margin = 2,
                latent_size = 128, #384, 
                n_gf = 64, 
                n_c=3):
        super().__init__()
        self.layers = nn.ModuleList()
        assert patch_size in [2,4,8,16]
        #n_z = latent_size + n_c * ( (patch_margin + patch_size)**2 - patch_size**2)
        n_z = latent_size + n_c * ( (2*patch_margin + patch_size)**2 )
        channel_coeff = {
                0:8, # 2x2
                1:4, # 4x4
                2:2, # 8x8
                3:1, # 16x16
                }
        kernel_size = {
                0:2, # 2x2
                1:4, # 4x4
                2:4, # 8x8
                3:4, # 16x16
                }
        stride = {
                0:1, # 2x2
                1:2, # 4x4
                2:2, # 8x8
                3:2, # 16x16
                }
        padding = {
                0:0, # 2x2
                1:1, # 4x4
                2:1, # 8x8
                3:1  # 16x16
                }
        
        self.main = nn.Sequential(
                nn.ConvTranspose2d(n_z, n_gf * 8, kernel_size=  2, bias=False),
#                nn.BatchNorm2d(n_gf * 8),
                nn.LeakyReLU(inplace=True),
                nn.ConvTranspose2d(n_gf * 8, n_gf * 8, kernel_size=  1, bias=False),
#                nn.BatchNorm2d(n_gf * 8),
                nn.LeakyReLU(inplace=True),
                )
        for layer_ind in range(1, int(math.log2(patch_size))): 
            self.main.append(
                    nn.ConvTranspose2d(
                        n_gf * channel_coeff[layer_ind-1], 
                        n_gf * channel_coeff[layer_ind], 
                        kernel_size= kernel_size[layer_ind],
                        stride= stride[layer_ind],
                        padding = padding[layer_ind],
                        bias= False
                        )
                    )
#            self.main.append(nn.BatchNorm2d(n_gf * channel_coeff[layer_ind]))
            self.main.append(nn.LeakyReLU(inplace=True))

        self.last = nn.Sequential(
                nn.Conv2d(n_gf * channel_coeff[int(math.log2(patch_size))-1], n_c, kernel_size=1, bias=True),
                nn.Sigmoid()
                )
                


    def forward(self, x, c):
        x = torch.cat([x.flatten(start_dim=1), c.flatten(start_dim=1)], dim= -1).unsqueeze(2).unsqueeze(3)
        return self.last(self.main(x))


class Discriminator(nn.Module):
    def __init__(self, 
            patch_size,
            patch_margin,
            n_df, 
            n_c=3):
        super().__init__()
        channel_coeff = {
                0:1, # 2x2
                1:2, # 4x4
                2:4, # 8x8
                3:8, # 16x16
                }
        self.main = nn.Sequential(
                nn.Conv2d(n_c, n_df, kernel_size =3, stride=2, padding=1, bias=False),
#                nn.BatchNorm2d(n_df),
                nn.LeakyReLU()
                )
        for layer_ind in range(1, math.ceil(math.log2(patch_size+ 2*patch_margin))): 
            self.main.append(
                    nn.Conv2d(
                        n_df * channel_coeff[layer_ind-1],
                        n_df * channel_coeff[layer_ind],
                        kernel_size = 3,
                        stride=2,
                        padding=1,
                        bias=False
                        )
                    )
#            self.main.append(nn.BatchNorm2d(n_df * channel_coeff[layer_ind]))
            self.main.append(nn.LeakyReLU())

        self.last = nn.Sequential(
                nn.Conv2d(n_df*channel_coeff[math.ceil(math.log2(patch_size+2*patch_margin))-1], 1, kernel_size = 1, stride = 1, padding = 0, bias=True),
                nn.Sigmoid(),
                ) 


    def forward(self, x):
        return self.last(self.main(x))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.normal_(m.bias.data, 0)
