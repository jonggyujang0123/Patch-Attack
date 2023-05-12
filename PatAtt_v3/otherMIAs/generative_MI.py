"""Import default libraries"""
import os

from utils.base_utils import set_random_seeds, get_accuracy, AverageMeter, WarmupCosineSchedule, WarmupLinearSchedule, load_ckpt, save_ckpt, get_data_loader
import torch.nn as nn
import torch
import wandb
import os
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from einops import rearrange
from torchvision import transforms
import itertools
from einops.layers.torch import Rearrange, Reduce
from otherMIAs.commun_GAN import Generator, Discriminator
"""import target classifier, generator, and discriminator """

import argparse

def para_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', 
                        type=int,
                        default=200)
    parser.add_argument('--batch_size',
                        type=int,
                        default=64)




    return args



def main():
    args = para_config()
    if args.wandb_active:
        wandb.init(project= args.wandb_project,
                   entity = args.wandb_id,
                   config = args,
                   name = f'Dataset: {args.dataset}',
                   group = f'Dataset: {args.dataset}')
    print(args)
    if not torch.cuda.is_available():
        raise ValueError("Should buy a GPU!")
    set_random_seeds(args.random_seed)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load target classifier
    if args.img_size == 32:
        from models.resnet_32x32 import resnet10 as resnet
        from models.resnet_32x32 import resnet50 as resnet_val
    else:
        from models.resnet import resnet18 as resnet
        from models.resnet import resnet50 as resnet_val

    classifier = resnet(
        num_classes=args.n_classes,
        num_channels=args.num_channel).to(args.device)
    classifier_val = resnet_val(
        num_classes=args.n_classes,
        num_channels=args.num_channel).to(args.device)

    if os.path.exists(args.ckpt_fpath):
        ckpt = load_ckpt(args.ckpt_fpath_class, is_best = True)
        classifier.load_state_dict(ckpt['model'])
        classifier.eval()
        print(f'{args.ckpt_fpath_class} model is loaded!')
    else:
        raise Exception('there is no generative checkpoint')

    if os.path.exists(args.ckpt_fpath_class_val):
        ckpt = load_ckpt(args.ckpt_fpath_class + '_val', is_best = True)
        classifier_val.load_state_dict(ckpt['model'])
        classifier_val.eval()
        print(f'{args.ckpt_fpath_class_val} model is loaded!')
    # load generator and discriminator
    G = Generator(args).to(args.device)
    D = Discriminator(args).to(args.device)
    if os.path.exists(args.ckpt_fpath_gen):
        ckpt = load_ckpt(args.ckpt_fpath_gen, is_best = True)
        G.load_state_dict(ckpt['model'])
        G.eval()
        print(f'{args.ckpt_fpath_gen} model is loaded!')
    else:
        raise Exception('there is no generative checkpoint')

    train(wandb, args, classifier, G, D)

def train(wandb,  args, classifier, G, D):
    Attacker = nn.Sequential(
            nn.Linear(args.n_classes, args.n_z),
            ).to(args.device)
    optimizer = optim.Adam(A.parameters(), lr=args.lr)
    if args.decay_type == 'cosine':
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.epoch * 100)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.epoch * 100)
    
    criterion = nn.CrossEntropyLoss()
    BCELoss = nn.BCELoss()
    # define matter
    Loss_attack = AverageMeter()
    Acc_total = AverageMeter()

    pbar = tqdm(
            range(args.epochs),
            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
            )
    for epoch in pbar:
        tepoch = tqdm(
                range(100),
                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                )
        for step in range(100):
            inputs = torch.eye(args.n_classes).to(args.device)
            optimizer.zero_grad()
            outputs = Attacker(inputs)
            images = G(outputs)
            discrim = D(images)
            attack_loss = criterion(outputs, torch.arange(args.n_classes).to(args.device))
            discrim_loss = BCELoss(discrim, torch.ones_like(discrim))
            loss = attack_loss + discrim_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            Loss_attack.update(attack_loss.item(), inputs.size(0))
            Acc_total.update(get_accuracy(outputs, torch.arange(args.n_classes).to(args.device)), inputs.size(0))
            tepoch.set_description(
                    f'Epoch: {epoch + 1}/{args.epochs} | '
                    f'Loss_attack: {Loss_attack.avg:.4f} | '
                    f'Acc_total: {Acc_total.avg:.4f} | '
                    )
        if epoch % 10 == 0:
            if args.wandb_active:
                wandb.log({
                    'Loss_attack': Loss_attack.avg,
                    'Acc_total': Acc_total.avg,
                    'epoch': epoch + 1,
                    })
                    



if __name__ == "__main__":
    main() 
