import os
from utils.base_utils import set_random_seeds, get_accuracy, AverageMeter, WarmupCosineSchedule, WarmupLinearSchedule, load_ckpt, save_ckpt, get_data_loader
import torch.nn as nn
import torch
import wandb
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange
from torchvision import transforms
import argparse
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch GAN')
    parser.add_argument('--dataset', 
                        default='HAN', 
                        type=str)
    parser.add_argument('--ckpt-path',
                        default='../experiments/common_gan',
                        type=str)
    parser.add_argument('--seed',
                        default=0,
                        type=int)
    parser.add_argument('--train-batch-size',
                        default=64,
                        type=int)
    parser.add_argument('--test-batch-size',
                        default=100,
                        type=int)
    parser.add_argument('--epochs',
                        default=50,
                        type=int)
    parser.add_argument('--lr',
                        default=0.0002,
                        type=float)
    parser.add_argument('--beta1',
                        default=0.5,
                        type=float)
    parser.add_argument('--beta2',
                        default=0.999,
                        type=float)
    parser.add_argument('--latent-size',
                        default=128,
                        type=int)
    parser.add_argument('--log-interval',
                        default=10,
                        type=int)
    parser.add_argument('--n-gf',
                        default=64,
                        type=int)
    parser.add_argument('--n-df',
                        default=64,
                        type=int)
    parser.add_argument('--levels',
                        default=3,
                        type=int)
    parser.add_argument('--decay-type',
                        default='cosine',
                        type=str)
    parser.add_argument('--warmup-steps',
                        default=100,
                        type=int)
    parser.add_argument('--num-workers',
                        default=8,
                        type=int)
    parser.add_argument('--pin-memory',
                        default=True,
                        type=bool)
    parser.add_argument('--wandb-active',
                        default=True,
                        type=bool)
    parser.add_argument('--wandb-project',
                        default='common_gan',
                        type=str)
    parser.add_argument('--wandb-id',
                        default='jonggyujang0123',
                        type=str)


    args = parser.parse_args()


    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.ckpt_fpath = os.path.join(args.ckpt_path, args.dataset)

    if args.dataset in 'HAN':
        args.num_channel = 1
        args.img_size = 32
    if args.dataset == 'mnist':
        args.num_channel = 1
        args.img_size = 32
    if args.dataset == 'cifar100':
        args.num_channel = 3
        args.img_size = 32
    if args.dataset == 'LFW':
        args.num_channel = 3
        args.img_size = 128
    return args

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            #  m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            #  m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.zero_()

class Generator(nn.Module):
    def __init__(self,
                 img_size = 32,
                 latent_size = 128,
                 n_gf = 64,
                 levels = 2,
                 n_c = 3,
                 ):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.latent_size = latent_size
        self.n_gf = n_gf
        self.levels = levels
        self.n_c = n_c

        self.init_size = self.img_size // 2**(levels)

        self.init_block = nn.Sequential(
                Rearrange('b c -> b c () ()'),
                nn.ConvTranspose2d(self.latent_size, self.n_gf * 2**levels, self.init_size, 1, 0),
                nn.BatchNorm2d(self.n_gf * 2**levels),
                nn.LeakyReLU(0.2, inplace=True),
                )
        self.deconv = nn.Sequential()
        for i in range(levels):
            self.deconv.add_module(
                    'deconv_{}'.format(i),
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            self.n_gf * 2**(levels-i),
                            self.n_gf * 2**(levels-i-1) if i < levels-1 else self.n_c,
                            4, 2, 1,
                            bias=False if i < levels-1 else True,
                            ),
                        nn.BatchNorm2d(self.n_gf * 2**(levels-i-1)) if i < levels-1 else nn.Identity(),
                        nn.LeakyReLU(0.2, inplace=True) if i < levels-1 else nn.Tanh(),
                        )
                    )
        self.apply(initialize_weights)
        #  initialize_weights(self)
    def forward(self, x):
        x = self.init_block(x)
        x = self.deconv(x)
        return x

def discriminator_block(in_filters, out_filters, bn=True):
    block = [
            nn.Conv2d(in_filters, out_filters, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(out_filters) if bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
            #  nn.Dropout2d(0.25),
            ]
    return block

class Discriminator(nn.Module):
    def __init__(
            self,
            img_size = 32,
            n_df = 64,
            n_c = 3,
            levels = 2,
            ):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.n_df = n_df
        self.n_c = n_c
        self.levels = levels
        
        self.conv = nn.Sequential()
        for i in range(levels):
            self.conv.add_module(
                f'conv_{i}',
                nn.Sequential(
                    *discriminator_block(
                        self.n_c if i == 0 else self.n_df * 2**(i-1),
                        self.n_df * 2**i,
                        bn = False if i == 0 else True,
                        ),
                    )
                )

        n_feat = self.conv(torch.zeros(1, self.n_c, self.img_size, self.img_size)).shape[2]

        self.fc = nn.Sequential(
                Rearrange('b c h w -> b (c h w)'),
                nn.Linear(n_feat**2 * n_df * 2**(levels-1), 1),
                nn.Sigmoid(),
                )
        self.apply(initialize_weights)
        #  initialize_weights(self)
    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)


def train(wandb, args, G, D):
    fixed_noise = torch.randn(args.test_batch_size, args.latent_size).to(args.device)
    BCE_loss = nn.BCELoss()
    optimizer_G = optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_D = optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    train_loader, _, _ = get_data_loader(args=args)
    Loss_g = AverageMeter()
    Loss_d = AverageMeter()
    D_x_total = AverageMeter()
    D_G_z_total = AverageMeter()
    pbar = tqdm(
            range(args.epochs),
            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
            )
    for epoch in pbar:
        G.train()
        D.train()
        tepoch = tqdm(
                train_loader,
                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                )
        for data in tepoch:
            # Train Discriminator
            optimizer_D.zero_grad()
            real_img = data[0].to(args.device)
            b_size = real_img.size(0)
            # Train Discriminator with real image
            output_real = D(real_img)
            d_loss_real = BCE_loss(output_real, torch.ones(b_size,1).to(args.device) - 0.2)
            # Train Discriminator with fake image
            z = torch.randn(b_size, args.latent_size).to(args.device)
            fake_img = G(z)
            output_fake = D(fake_img.detach())
            d_loss_fake = BCE_loss(output_fake, torch.zeros(b_size,1).to(args.device))
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()
            # Train Generator
            optimizer_G.zero_grad()
            g_loss = BCE_loss(D(fake_img), torch.ones(b_size,1).to(args.device))
            g_loss.backward()
            optimizer_G.step()
            # Update meters
            D_x = output_real.mean().item()
            D_G_z = output_fake.mean().item()
            Loss_g.update(g_loss.item(), b_size)
            Loss_d.update(d_loss.item(), b_size)
            D_x_total.update(D_x, b_size)
            D_G_z_total.update(D_G_z, b_size)

            tepoch.set_description(
                    f'Epoch [{epoch+1}/{args.epochs}] '
                    f'Loss_d: {Loss_d.val:.4f}, '
                    f'Loss_g: {Loss_g.val:.4f}, '
                    f'D(x): {D_x_total.val:.4f}, '
                    f'D(G(z)): {D_G_z_total.val:.4f},'
                    f'Lr : {args.lr:.4f}'
                    )
            tepoch.refresh()
        Loss_g.reset()
        Loss_d.reset()
        D_x_total.reset()
        D_G_z_total.reset()

        if epoch % args.log_interval == args.log_interval-1:
            G.eval()
            test_img = G(fixed_noise)
            test_img = test_img.reshape(10, 10, args.num_channel,args.img_size,args.img_size)
            if args.num_channel == 1:
                test_img = F.pad(test_img, (1,1,1,1), value=1)
            else:
                test_img = F.pad(test_img, (1,1,1,1), value=-1)
            image_grid = rearrange(test_img,
                                   'b1 b2 c h w -> (b1 h) (b2 w) c').cpu().detach().numpy().astype(np.float64)
            image_grid = wandb.Image(image_grid, caption=f'Epoch {epoch+1}')

            wandb.log({
                'Loss_g': Loss_g.avg,
                'Loss_d': Loss_d.avg,
                'D(x)': D_x_total.avg,
                'D(G(z))': D_G_z_total.avg,
                'image': image_grid,
                }, step=epoch+1)
            model_to_save_g = G.module if hasattr(G, 'module') else G
            model_to_save_d = D.module if hasattr(D, 'module') else D
            ckpt = {
                    'model_g' : model_to_save_g.state_dict(),
                    'model_d' : model_to_save_d.state_dict(),
                    }
            save_ckpt(checkpoint_fpath = args.ckpt_fpath, checkpoint = ckpt, is_best = True)

def main():
    args = parse_args()
    print(args)
    if args.wandb_active:
        wandb.init(project = args.wandb_project, 
                   entity = args.wandb_id,
                   config = args,
                   name = f'{args.dataset}',
                   group = f'{args.dataset}')
    else:
        os.environ["WANDB_SILENT"] = "true"
    set_random_seeds(args.seed)

    
    G = Generator(
            img_size = args.img_size,
            latent_size = args.latent_size,
            n_gf = args.n_gf,
            levels = args.levels,
            n_c = args.num_channel,
            ).to(args.device)

    D = Discriminator(
            img_size = args.img_size,
            n_df = args.n_df,
            n_c = args.num_channel,
            levels = args.levels,
            ).to(args.device)
    train(wandb, args, G, D)


if __name__ == '__main__':
    main()
