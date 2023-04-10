"""
mnist tutorial main model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
import numpy as np
def normal_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    

class Generator(nn.Module):
    def __init__(self, 
                img_size=32,
                n_classes = 10,
                latent_size = 128, #384, 
                n_gf = 64,
                levels = 2,
                n_c=3
                ):
        super().__init__()
        self.init_size = img_size // (2**levels) 
        self.layers = nn.ModuleList()
        self.n_classes = n_classes
        assert img_size in [32, 64, 128]
        #levels = int(math.log2(img_size)) - 2# if image size is 32, number of levels is 3 {0, 1, 2} with channel coefficients of  {1, 2, 4}
        
        self.label_layer = nn.Embedding(n_classes, latent_size)
#        self.label_layer = nn.Sequential(
#                nn.ConvTranspose2d(n_classes, n_gf * 2**(levels), self.init_size, 1, 0),
#                nn.BatchNorm2d(n_gf * 2**(levels)),
#                nn.LeakyReLU(0.2, inplace=True),
#                )
        self.latent_layer = nn.Sequential(
                nn.ConvTranspose2d(latent_size, n_gf * 2**(levels), self.init_size, 1, 0),
                nn.BatchNorm2d(n_gf * 2**(levels)),
                nn.LeakyReLU(0.2, inplace=True),
                )
#        self.latent_generator = nn.Sequential(
#                nn.Linear(n_z, n_z),
#                nn.BatchNorm1d(n_z),
#                nn.LeakyReLU(inplace=True),
#                )
        self.main = nn.Sequential()
        self.main.append(
                nn.BatchNorm2d(n_gf * 2**(levels))
                )
        for layer_ind in range(levels):
            block = nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(
                        n_gf * 2 **(levels - layer_ind), n_gf * 2 **(levels - 1 - layer_ind), 3, stride=1, padding=1),
                    nn.BatchNorm2d(n_gf* 2**(levels-1-layer_ind), 0.8),
                    nn.LeakyReLU(0.2,inplace=True)
                    )
            self.main.append(block)
#            self.main.add_module(name=f'block{layer_ind}', module=block)

        self.main.append(
                nn.Sequential(
                    nn.Conv2d(n_gf, n_c, 3, stride=1, padding=1),
                    nn.Tanh()
                    )
                )

    def weight_init(self):
        for m in self._modules:
            normal_init(self._modules[m])

    def forward(self, x, c):
#        c = self.label_layer(F.one_hot(c, num_classes = self.n_classes).float().unsqueeze(2).unsqueeze(3)) 
        c = self.label_layer(c) 
        x = self.latent_layer((x*c).unsqueeze(2).unsqueeze(3))
#        x = self.latent_layer(x.unsqueeze(2).unsqueeze(3))
#        print(x)
#        x = self.latent_generator(x)
#        x = (x+c).unsqueeze(2).unsqueeze(3)
#        x = torch.cat( (x, c), dim=1).unsqueeze(2).unsqueeze(3)
#        x = torch.cat( (x, self.label_cond_generator(c)), dim=1).unsqueeze(2).unsqueeze(3)
#        x = torch.cat( (self.latent_generator(x), self.label_cond_generator(c)), dim=1).unsqueeze(2).unsqueeze(3)
        return self.main(x)
#        return self.main(torch.cat([x, c], dim=1))


class Discriminator(nn.Module):
    def __init__(self, 
            patch_size,
            levels=4,
            n_df = 16,
            n_c=3):
        super().__init__()
        self.bs = patch_size // 2 ** levels

        def discriminator_block(in_filters, out_filters, bn = True):
            block = nn.Sequential(nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(0.25))
            if bn:
                block.append(nn.BatchNorm2d(out_filters,0.8))
            return block

        self.main = nn.Sequential(
                discriminator_block(n_c, n_df, bn=False))
        for layer_ind in range(1,levels):
            self.main.append(
                    discriminator_block(n_df * 2 **(layer_ind - 1), n_df * 2 ** (layer_ind))
                    )
        self.main.append(
                nn.Sequential(
                    nn.Flatten(start_dim = 1, end_dim=-1),
                    nn.Linear(self.bs ** 2 * n_df * 2 ** (levels-1), 1),
                    nn.Sigmoid()
                    )
                )

    def forward(self, x):
#        return self.last(self.main(x))
        return self.main(x)
#        return self.last(self.main(x))

    def weight_init(self):
        for m in self._modules:
            normal_init(self._modules[m])
