"""
mnist tutorial main model
"""
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, 
                patch_size=4,
                n_z = 128, #384, 
                n_gf = 64, 
                n_c=1):
        super().__init__()
        self.layers = nn.ModuleList()
        assert patch_size in [4,8]
        self.channel_coeff = {
                1:8, # 2x2
                2:8, # 2x2
                3:4, # 4x4
                4:4, # 4x4
                5:2, # 8x8
                6:2  # 8x8
                }
        self.kernel_size = {
                1:2, # 2x2
                2:2, # 2x2
                3:4, # 4x4
                4:4, # 4x4
                5:4, # 8x8
                6:4  # 8x8
                }
        self.stride = {
                1:1, # 2x2
                2:1, # 2x2
                3:2, # 4x4
                4:2, # 4x4
                5:2, # 8x8
                6:2  # 8x8
                }
        self.padding = {
                1:0, # 2x2
                2:1, # 2x2
                3:2, # 4x4
                4:2, # 4x4
                5:2, # 8x8
                6:2  # 8x8
                }
                

        self.main = nn.Sequential(
                #Layer 1
                nn.ConvTranspose2d(n_z, n_gf * 8, kernel_size=4, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(n_gf*8),
                nn.ReLU(inplace=True),
                #Layer 2
                nn.ConvTranspose2d(n_gf * 8 , n_gf * 4, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(n_gf*4),
                nn.ReLU(inplace=True),
                #Layer 3
                nn.ConvTranspose2d(n_gf * 4, n_gf * 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(n_gf*2),
                nn.ReLU(inplace=True),
                #Layer 2
                nn.ConvTranspose2d(n_gf * 2, n_gf, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(n_gf),
                nn.ReLU(inplace=True),
                #Layer 2
                nn.ConvTranspose2d(n_gf, n_c, kernel_size=4, stride=2, padding=1, bias=False),
                nn.Tanh(),
                )


    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, n_df, n_c=3):
        super().__init__()
        self.main = nn.Sequential(
                #Layer 1
                nn.Conv2d(n_c, n_df, kernel_size = 4, stride = 2 ,padding =1, bias=False),
                nn.LeakyReLU(0.2),
                #Layer 2
                nn.Conv2d(n_df, n_df*2, kernel_size = 4, stride = 2 ,padding =1, bias=False),
                nn.BatchNorm2d(n_df*2),
                nn.LeakyReLU(0.2),
                #Layer 3
                nn.Conv2d(n_df*2, n_df*4, kernel_size = 4, stride = 2, padding = 1, bias=False),
                nn.BatchNorm2d(n_df*4),
                nn.LeakyReLU(0.2),
                #Layer 4
                nn.Conv2d(n_df*4, n_df*8, kernel_size = 4, stride = 2, padding = 1, bias=False),
                nn.BatchNorm2d(n_df*8),
                nn.LeakyReLU(0.2),
                #Layer 5
                nn.Conv2d(n_df*8, 1, kernel_size = 4, stride = 1, padding = 0, bias=False),
                nn.Sigmoid(),
                ) 


    def forward(self, x):
        return self.main(x)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.normal_(m.bias.data, 0)
