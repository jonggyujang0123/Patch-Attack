"""
mnist tutorial main model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
import numpy as np
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class Generator(nn.Module):
    def __init__(self, 
                img_size=32,
                n_classes = 10,
                latent_size = 128, #384, 
                n_gf = 64, 
                n_c=3
                ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.n_classes = n_classes
        assert img_size in [32, 64, 128]
        levels = int(math.log2(img_size)) - 2# if image size is 32, number of levels is 3 {0, 1, 2} with channel coefficients of  {1, 2, 4}

        self.label_layer = nn.Sequential(
                nn.ConvTranspose2d(n_classes, 10, 4, 1, 0),
                nn.BatchNorm2d(10),
                nn.LeakyReLU(0.2, inplace=True),
                )
        self.latent_layer = nn.Sequential(
                nn.ConvTranspose2d(latent_size, n_gf * 2**(levels -1) -10, 4, 1, 0),
                nn.BatchNorm2d(n_gf * 2**(levels-1) -10),
                nn.LeakyReLU(0.2, inplace=True),
                )
#        self.latent_generator = nn.Sequential(
#                nn.Linear(n_z, n_z),
#                nn.BatchNorm1d(n_z),
#                nn.LeakyReLU(inplace=True),
#                )
        self.main = nn.Sequential()
        for layer_ind in range(levels):
            self.main.append(
                    nn.ConvTranspose2d(
                        n_gf * 2**(levels-1-layer_ind),
                        n_gf * 2**(levels-2-layer_ind) if layer_ind < levels-1 else n_c,
                        4,
                        2,
                        1
                        )
                    )
            if layer_ind < levels-1:
                self.main.append(nn.BatchNorm2d(n_gf* 2**(levels-2-layer_ind)))
#                self.main.append(nn.ReLU(True))
                self.main.append(nn.LeakyReLU(0.2,inplace=True))
            else:
                self.main.append(nn.Tanh())

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x, c):
        c = self.label_layer(F.one_hot(c, num_classes = self.n_classes).float().unsqueeze(2).unsqueeze(3)) 
        x = self.latent_layer(x.unsqueeze(2).unsqueeze(3))
#        print(x)
#        x = self.latent_generator(x)
#        x = (x+c).unsqueeze(2).unsqueeze(3)
#        x = torch.cat( (x, c), dim=1).unsqueeze(2).unsqueeze(3)
#        x = torch.cat( (x, self.label_cond_generator(c)), dim=1).unsqueeze(2).unsqueeze(3)
#        x = torch.cat( (self.latent_generator(x), self.label_cond_generator(c)), dim=1).unsqueeze(2).unsqueeze(3)
        return self.main(torch.cat([x, c], dim=1))


class Discriminator(nn.Module):
    def __init__(self, 
            patch_size,
            n_df = 64,
            n_c=3):
        super().__init__()
        channel_coeff = {
                0:1, # 2x2
                1:2, # 4x4
                2:4, # 8x8
                3:4, # 16x16
                4:4, # 32x32
                }
        base_size = int(2**(np.ceil(np.log2(patch_size**2 * n_c))+1))
        self.main_fc = nn.Sequential(
                nn.Flatten(start_dim=1, end_dim=-1),
                nn.Linear(patch_size **2 * n_c, base_size),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(base_size, base_size),
                nn.Dropout(0.3),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(base_size, base_size),
                nn.Dropout(0.3),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(base_size, 1),
                nn.Sigmoid(),
                )
        self.main = nn.Sequential(
                nn.Conv2d(n_c, n_df, kernel_size =4, stride=2, padding=1, bias=True),
                nn.BatchNorm2d(n_df),
                nn.LeakyReLU()
                )
        for layer_ind in range(1, math.ceil(math.log2(patch_size))): 
            self.main.append(
                    nn.Conv2d(
                        n_df * channel_coeff[layer_ind-1],
                        n_df * channel_coeff[layer_ind],
                        kernel_size = 4,
                        stride=2,
                        padding=1,
                        bias=True
                        )
                    )
            if layer_ind < math.ceil(math.log2(patch_size))-1:
                self.main.append(nn.BatchNorm2d(n_df * channel_coeff[layer_ind]))
            self.main.append(nn.LeakyReLU())

        self.last = nn.Sequential(
                nn.Conv2d(n_df*channel_coeff[math.ceil(math.log2(patch_size))-1], n_df*channel_coeff[math.ceil(math.log2(patch_size))-1], kernel_size = 1, stride = 1, padding = 0, bias=True),
                nn.BatchNorm2d(n_df*channel_coeff[math.ceil(math.log2(patch_size))-1]),
                nn.LeakyReLU(),
                nn.Conv2d(n_df*channel_coeff[math.ceil(math.log2(patch_size))-1], 1, kernel_size = 1, stride = 1, padding = 0, bias=True),
                nn.Sigmoid(),
                ) 


    def forward(self, x):
        return self.last(self.main(x))
#        return self.main_fc(x)
#        return self.last(self.main(x))

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
