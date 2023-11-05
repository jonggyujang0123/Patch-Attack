"""Import default libraries"""
import os

from utils.base_utils import set_random_seeds, get_accuracy, AverageMeter, WarmupCosineSchedule, WarmupLinearSchedule, load_ckpt, save_ckpt, get_data_loader, accuracy
import torch.nn as nn
import torch
import wandb
import os
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from otherMIAs.common_GAN import Generator, Discriminator
from torch.nn import functional as F
from models.resnet_32x32 import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from models.densenet import DenseNet121, DenseNet161, DenseNet169, DenseNet201
from models.dla import DLA
"""import target classifier, generator, and discriminator """

import argparse

def para_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',
                        type=int,
                        default=1000)
    parser.add_argument('--n-images',
                        type=int,
                        default=1000)
    parser.add_argument('--random-seed',
                        type=int,
                        default=0)
    parser.add_argument('--lr',
                        type=float,
                        default=3e-1)
    parser.add_argument('--n-gf',
                        type=int,
                        default=64)
    parser.add_argument('--n-df',
                        type=int,
                        default=64)
    parser.add_argument('--latent-size',
                        type=int,
                        default=128)
    parser.add_argument('--levels',
                        type=int,
                        default=3)
    parser.add_argument('--w-gan',
                        type=float,
                        default=0.15)
    parser.add_argument('--target-dataset',
                        type=str,
                        default='mnist')
    parser.add_argument('--aux-dataset',
                        type=str,
                        default='HAN')
    parser.add_argument('--wandb-active',
                        type=bool,
                        default=True)
    parser.add_argument('--wandb-project',
                        type=str,
                        default='generative_MI')
    parser.add_argument('--wandb-id',
                        type=str,
                        default='jonggyujang0123')
    parser.add_argument('--eval-every',
                        type=int,
                        default=50)
    parser.add_argument('--num-workers',
                        type=int,
                        default=4)
    parser.add_argument('--pin-memory',
                        type=bool,
                        default=True)
    parser.add_argument('--ckpt-dir',
                        type=str,
                        default='../experiments')
    parser.add_argument('--target-class',
                        type=int,
                        default=0)
    parser.add_argument('--beta1',
                        type=float,
                        default=0.1)
    parser.add_argument('--beta2',
                        type=float,
                        default=0.1)
    parser.add_argument('--weight-decay',
                        type=float,
                        default=5e-4)



    args = parser.parse_args()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.ckpt_path = os.path.join(args.ckpt_dir, 'classifier', args.target_dataset)
    args.ckpt_path_gan = os.path.join(args.ckpt_dir, 'common_gan', args.aux_dataset)
    if args.target_dataset == 'mnist':
        args.num_channel = 1
        args.img_size = 32
        args.num_classes = 10
    elif args.target_dataset == 'emnist':
        args.num_channel = 1
        args.img_size = 32
        args.num_classes = 26
    elif args.target_dataset == 'cifar10':
        args.num_channel = 3
        args.img_size = 32
        args.num_classes = 10
    else:
        raise NotImplementedError
    return args



def train(wandb,  args, classifier, classifier_val, G, D):
    Attacker = torch.autograd.Variable(
            torch.randn(args.n_images,
                        args.latent_size,
                        ).to(args.device),
            requires_grad=True,
            )
    optimizer = optim.SGD([Attacker], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    #  optimizer = optim.Adam([Attacker], lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
    #  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs*0.5), int(args.epochs * 0.75)], gamma=0.3)
    # define matter
    #  BCELoss = nn.BCELoss().to(args.device)
    BCELoss = nn.BCEWithLogitsLoss().to(args.device)
    Total_attack_loss = AverageMeter()
    Total_loss = AverageMeter()
    Acc_total = AverageMeter()

    pbar = tqdm(
            range(args.epochs),
            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
            )
    scaler = torch.cuda.amp.GradScaler()

    for epoch in pbar:
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            img = G(Attacker)
            logits = classifier(img)
            attack_loss = - logits.softmax(1)[:, args.target_class].log().mean()
            d_logits = D(img)
            discrim_loss = BCELoss(d_logits, torch.ones_like(d_logits).to(args.device))
            loss = attack_loss + args.w_gan * discrim_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        #  loss.backward()
        #  optimizer.step()
        
        train_acc = logits.argmax(1).eq(args.target_class).float().mean().item()
        Acc_total.update(train_acc, 1)
        Total_attack_loss.update(attack_loss.item(), 1)
        Total_loss.update(loss.item(), 1)
        scheduler.step()
        tqdm.write(
                f'Epoch: {epoch + 1}/{args.epochs} | '
                f'Loss_total: {Total_loss.avg:.4f} | '
                f'Loss_attack: {Total_attack_loss.avg:.4f} | '
                f'Acc_total: {Acc_total.avg:.4f} | '
                f'lr: {scheduler.get_lr()[0]:.4f}'
                )
        if epoch % args.eval_every == 0:
            images, val_acc = evaluate(wandb, args, classifier_val, Attacker, G)
            if args.wandb_active:
                wandb.log({
                    'Loss_attack': Total_attack_loss.avg,
                    'Loss_total': Total_loss.avg,
                    'Acc_total': Acc_total.avg,
                    'Acc_val': val_acc,
                    'images' : images 
                    },
                    step=epoch + 1)
    return Attacker

def evaluate(wandb, args, classifier_val, Attacker, G):
    classifier_val.eval()
    img = G(Attacker)
    logits = classifier_val(img)
    val_acc = logits.argmax(1).eq(args.target_class).float().mean().item()
    if args.num_channel == 1:
        fake = F.pad(img, pad = (1,1,1,1), value = 1)
    else:
        fake = F.pad(img, pad = (1,1,1,1), value = -1)
    image_array = rearrange(fake[:100], 
                '(b1 b2) c h w -> (b1 h) (b2 w) c', b1 = 10, b2 = 10).cpu().detach().numpy().astype(np.float64)
    images = wandb.Image(image_array, caption=f'Acc is {val_acc*100:2.2f}')
    return images, val_acc


def save_images(args, Attacker, G):
    from torchvision.utils import save_image
    import torchvision

    fake = (G(Attacker) + 1) / 2
    directory = f'./Results/Generative_MI/{args.target_dataset}/{args.target_class}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i in range(args.n_images):
        tensor = fake[i, ...].cpu().detach()
        if args.num_channel == 1:
            tensor = torch.cat([tensor, tensor, tensor], dim = 0)
        save_image(tensor, f'{directory}/{i}.png')

def main():
    args = para_config()
    if args.wandb_active:
        wandb.init(project= args.wandb_project,
                   entity = args.wandb_id,
                   config = args,
                   name = f'Dataset: {args.target_dataset} | Class: {args.target_class}',
                   group = f'Dataset: {args.target_dataset}')
    else:
        os.environ['WANDB_MODE'] = 'dryrun'
    print(args)
    set_random_seeds(args.random_seed)

    # load target classifier
    classifier = ResNet18(num_classes = args.num_classes, num_channel = args.num_channel).to(args.device)
    classifier_val = DLA(num_classes = args.num_classes, num_channel = args.num_channel).to(args.device)

    if os.path.exists(args.ckpt_path):
        ckpt = load_ckpt(args.ckpt_path, is_best = True)
        classifier.load_state_dict(ckpt['model'])
        classifier.eval()
        print(f'{args.ckpt_path} model is loaded!')
    else:
        raise ValueError(f'{args.ckpt_path} does not exist!')

    if os.path.exists(args.ckpt_path + '_valid'):
        ckpt = load_ckpt(args.ckpt_path + '_valid', is_best = True)
        classifier_val.load_state_dict(ckpt['model'])
        classifier_val.eval()
        print(f'{args.ckpt_path}_valid model is loaded!')
    else:
        raise ValueError(f'{args.ckpt_path}_valid does not exist!')

    # load generator and discriminator
    G = Generator(
            img_size=args.img_size,
            latent_size = args.latent_size,
            n_gf = args.n_gf,
            levels = args.levels,
            n_c = args.num_channel
            ).to(args.device)
    D = Discriminator(
            img_size=args.img_size,
            n_df = args.n_df,
            levels = args.levels,
            n_c = args.num_channel
            ).to(args.device)
    if os.path.exists(args.ckpt_path_gan):
        ckpt = load_ckpt(args.ckpt_path_gan)
        G.load_state_dict(ckpt['model_g'])
        G.eval()
        D.load_state_dict(ckpt['model_d'])
        D.eval()
        print(f'{args.ckpt_path_gan} model is loaded!')
    else:
        raise Exception('there is no generative checkpoint')

    Attacker = train(wandb, args, classifier, classifier_val, G, D)
    save_images(args, Attacker, G)



if __name__ == "__main__":
    main() 
