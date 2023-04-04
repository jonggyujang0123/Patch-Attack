"""
mnist tutorial main model
"""
import math 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import Pad 
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
        self.weight_init()

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
            img_size=32,
            patch_size=6,
            patch_stride=2,
            levels=4,
            n_df = 16,
            n_c=3,
            batch_size = 64):
        super().__init__()
        self.pad = Pad(patch_size - patch_stride, fill=0)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride= patch_stride
        self.bs = math.ceil(patch_size / (2 ** levels) )
        self.n_patches = ((img_size + 2* (patch_size - patch_stride) -patch_size) // patch_stride + 1) **2
#        pos_emb = positionalencoding2d(n_df, int(self.n_patches**0.5), int(self.n_patches**0.5) ).permute(2, 0, 1).reshape([1, self.n_patches, n_df, 1, 1]).tile([batch_size, 1, 1, 1, 1])

        pos_emb = nn.Parameter(torch.randn( 1, self.n_patches, n_df, 1, 1 )).tile([batch_size, 1, 1, 1, 1])
        self.pos_emb1D = pos_emb.reshape(pos_emb.shape[0] * pos_emb.shape[1], pos_emb.shape[2], 1, 1)

        def discriminator_block(in_filters, out_filters, bn = True):
            block = nn.Sequential(nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(0.25))
            if bn:
                block.append(nn.BatchNorm2d(out_filters,0.8))
            return block

        self.main_1 = nn.Sequential(
                discriminator_block(n_c, n_df, bn=False))
        self.main_2 = nn.Sequential()
        for layer_ind in range(1,levels):
            self.main_2.append(
                    discriminator_block(n_df * 2 **(layer_ind - 1), n_df * 2 ** (layer_ind))
                    )
        self.main_2.append(
                nn.Sequential(
                    nn.Flatten(start_dim = 1, end_dim=-1),
                    nn.Linear(self.bs ** 2 * n_df * 2 ** (levels-1), 1),
                    nn.Sigmoid()
                    )
                )


        self.weight_init()

    def forward(self, x):
        x = self.pad(x).unfold(2, self.patch_size, self.patch_stride).unfold(3, self.patch_size, self.patch_stride)
#        print(x.shape)
        x = x.contiguous().view(
                x.shape[0], x.shape[2] * x.shape[3], x.shape[1], x.shape[4], x.shape[5]
                ) # [b, n_pat, c, p_size, p_size]
        x = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        # index 
#        print(self.n_patches)

        rand_ind = np.random.choice(np.arange(x.shape[0]), x.shape[0]//30)
        x = x[rand_ind, :, :, :]
        pos_emb = self.pos_emb1D[rand_ind, :, :, :].to(x.device)
        x = self.main_1(x)
        x = x + pos_emb
#        return self.last(self.main(x))
        return self.main_2(x)
#        return self.last(self.main(x))

    def weight_init(self):
        for m in self._modules:
            normal_init(self._modules[m])





def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe
