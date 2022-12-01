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
        emb_size = self.channel * (3 * patch_margin ** 2 + 2 * patch_margin * patch_size)
        if activation_fn == 'relu':
            activ = nn.ReLU()
        elif activation_fn == 'leakyrelu':
            activ = nn.LeakyReLU()
        else:
            raise Exception("unknown actiation function!")

        self.encoder = nn.Sequential(
                nn.Linear(self.channel * patch_size**2 + emb_size, latent_size*8),
                activ,
                nn.Linear(latent_size*8, latent_size*8),
                activ,
                nn.Linear(latent_size*8, latent_size*2)
                )
        self.decoder = nn.Sequential(
                nn.Linear(latent_size + emb_size, 100),
                activ,
                nn.Linear(100, 100),
                activ,
                nn.Linear(100, self.channel * patch_size **2),
                nn.Sigmoid()
                )
        self.embedder = nn.Embedding((img_size//patch_size)**2, pos_emb_size)
        self.training = training

        
        self.pe = torch.zeros(self.num_patches, pos_emb_size)
        position = torch.arange(0, self.num_patches).unsqueeze(1)
        div_term = torch.exp(
                torch.arange(0, pos_emb_size, 2, dtype=torch.float) * -(math.log(400.0) / pos_emb_size)
                )
        self.pe[:, 0::2] = torch.sin(position.float() * div_term)
        self.pe[:, 1::2] = torch.cos(position.float() * div_term)

    def emb(self, pos):
        with torch.no_grad():
            embeddings = self.embedder(pos)

        return embeddings
    
    def pos_emb(self, pos):
        return self.pe[pos,:].to(pos.device)

    def conditioning(self, cond, pos):
        with torch.no_grad():
            pos_emb_vec = self.pos_emb(pos)
        cond_freeze = cond.flatten(start_dim=1) #self.conditioner(cond.flatten(start_dim=1))
        cond = cond_freeze #self.conditioner(cond.flatten(start_dim=1))
        #cond = torch.cat([cond,pos_emb_vec], dim=1)
        #cond_freeze = torch.cat([cond_freeze,pos_emb_vec], dim=1)
        #cond_freeze = torch.cat([cond_freeze,pos_emb_vec], dim=1)
        return cond, cond_freeze


    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mean + std * eps if self.training else mean

    def encode(self, x, cond, pos):
        cond, _ = self.conditioning( cond, pos)
        mean_var = self.encoder(torch.cat([x.flatten(start_dim=1),cond], dim=1))
        #mean_var = self.encoder(x.flatten(start_dim=1))
        mean = mean_var[:,:self.latent_size]
        log_var = mean_var[:,self.latent_size::]
        
        return mean, log_var 

    def decode(self, z, cond, pos):
        _, cond_freeze = self.conditioning( cond, pos)
        return self.decoder(torch.cat([z, cond_freeze], dim=1)).reshape([-1, self.channel, self.patch_size, self.patch_size])

    def forward(self, x, cond, pos):
        """
        x : target patch image (bs, channel, patch_size, patch_size)
        cond : target patch image (bs, patch_size * 2, patch_size * 2) in values (0~1 : pixels, -1: not painted)
        pos : int 
        """
        mean, log_var = self.encode(x, cond, pos)
        z = self.reparameterize(mean, log_var)
        return self.decode(z, cond, pos), mean, log_var

