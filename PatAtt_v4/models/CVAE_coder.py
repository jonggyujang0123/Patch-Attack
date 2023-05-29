import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
class View(nn.Module):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

class CVAE(nn.Module):
    def __init__(
            self,
            img_size = 32,
            latent_size = 64,
            patch_size = 4,
            patch_margin = 2,
            emb_size = 36,
            pos_emb_size = 2,
            grayscale=True,
            training=True,
            activation_fn = 'relu'
            ):
        super(CVAE, self).__init__()
        self.channel = 1 if grayscale else 3
        self.latent_size = latent_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size)**2
        #emb_size = self.channel * (3 * patch_margin ** 2 + 2 * patch_margin * patch_size)
        emb_size = self.channel * (patch_margin *2  + patch_size) **2
        if activation_fn == 'relu':
            activ = nn.ReLU()
        elif activation_fn == 'leakyrelu':
            activ = nn.LeakyReLU()
        else:
            raise Exception("unknown actiation function!")

        self.encoder = nn.Sequential(
#                nn.Conv2d(self.channel, 64, 
                nn.Linear(self.channel * patch_size**2 + emb_size, latent_size*8),
                activ,
                nn.Linear(latent_size*8, latent_size*8),
                activ,
                nn.Linear(latent_size*8, latent_size*8),
                activ,
                nn.Linear(latent_size*8, latent_size*2)
                )
#        self.encoder_CNN_1 = nn.Sequential(
#                nn.Linear(self.channel * patch_size**2 + emb_size, latent_size*8),
#                activ,
#                nn.Linear(latent_size*8, latent_size*8),
#                activ,
#                nn.Linear(latent_size*8, latent_size*8),
#                activ,
#                nn.Linear(latent_size*8, latent_size*2)
#                )
#        self.encoder_CNN_2 = nn.Sequential(
#                nn.Linear(self.channel * patch_size**2 + emb_size, latent_size*8),
#                activ,
#                nn.Linear(latent_size*8, latent_size*8),
#                activ,
#                nn.Linear(latent_size*8, latent_size*8),
#                activ,
#                nn.Linear(latent_size*8, latent_size*2)
#                )
        self.decoder = nn.Sequential(
                nn.Linear(latent_size, 256),
                activ,
                nn.Linear(256, 256),
                activ,
                nn.Linear(256, 256),
                activ,
                nn.Linear(256, 256),
                activ,
                )
        self.decoder_cond = nn.Sequential(
                nn.Linear(256 + emb_size, 256),
                activ,
                nn.Linear(256, 256),
                activ,
                nn.Linear(256, self.channel * patch_size **2),
                nn.Sigmoid()
                )
        self.training = training

        

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mean + std * eps if self.training else mean

    def encode(self, x, cond):
        cond = cond.flatten(start_dim=1)
        cond = cond*0
        mean_var = self.encoder(torch.cat([x.flatten(start_dim=1),cond], dim=1))
        mean = mean_var[:,:self.latent_size]
        log_var = mean_var[:,self.latent_size::]
        
        return mean, log_var 

    def decode(self, z, cond):
        z = self.decoder(z)
#        cond = cond*0
        cond = cond.flatten(start_dim=1)
        return self.decoder_cond(torch.cat([z, cond], dim=1)).reshape([-1, self.channel, self.patch_size, self.patch_size])

    def forward(self, x, cond):
        """
        x : target patch image (bs, channel, patch_size, patch_size)
        cond : target patch image (bs, patch_size * 2, patch_size * 2) in values (0~1 : pixels, -1: not painted)
        """
        mean, log_var = self.encode(x, cond )
#        cond = cond*0
        z = self.reparameterize(mean, log_var)
        return self.decode(z, cond), mean, log_var

