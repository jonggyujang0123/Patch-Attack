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
from otherMIAs.common_GAN import Generator, Discriminator
"""import target classifier, generator, and discriminator """

import argparse

def para_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',
                        type=int,
                        default=50)
    parser.add_argument('--batch-size',
                        type=int,
                        default=64)
    parser.add_argument('--random-seed',
                        type=int,
                        default=0)
    parser.add_argument('--lr',
                        type=float,
                        default=3e-2)
    parser.add_argument('--n-gf',
                        type=int,
                        default=64)
    parser.add_argument('--n-df',
                        type=int,
                        default=64)
    parser.add_argument('--levels',
                        type=int,
                        default=2)
    parser.add_argument('--decay-type',
                        type=str,
                        default='cosine')
    parser.add_argument('--warmup-steps',
                        type=int,
                        default=100)
    parser.add_argument('--n-classes',
                        type=int,
                        default=10)
    parser.add_argument('--latent-size',
                        type=int,
                        default=64)
    parser.add_argument('--w-gan',
                        type=float,
                        default=0.15)
    parser.add_argument('--img-size',
                        type=int,
                        default=32)
    parser.add_argument('--num-channel',
                        type=int,
                        default=1)
    parser.add_argument('--target-dataset',
                        type=str,
                        default='mnist')
    parser.add_argument('--aux-dataset',
                        type=str,
                        default='emnist')
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
                        default=10)
    parser.add_argument('--ckpt-dir',
                        type=str,
                        default='../experiments')


    args = parser.parse_args()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.ckpt_path = os.path.join(args.ckpt_dir, 'classifier', args.target_dataset)
    args.ckpt_path_gan = os.path.join(args.ckpt_dir, 'common_gan', args.aux_dataset)
    return args



def train(wandb,  args, classifier, classifier_val, G, D):
    Attacker = nn.Sequential(
            nn.Linear(args.n_classes, args.latent_size),
            ).to(args.device)
    optimizer = optim.Adam(Attacker.parameters(), lr=args.lr)
    if args.decay_type == 'cosine':
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.epochs * 100)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.epochs * 100)
    
    criterion = nn.CrossEntropyLoss()
    # define matter
    BCELoss = nn.BCELoss()
    Total_attack_loss = AverageMeter()
    Total_loss = AverageMeter()
    Acc_total = AverageMeter()

    pbar = tqdm(
            range(args.epochs),
            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
            )
    for epoch in pbar:
        for _ in range(100):
            inputs = torch.eye(args.n_classes).to(args.device).float()
            optimizer.zero_grad()
            outputs = Attacker(inputs)
            images = G(outputs)
            discrim = D(images)
            output_classifier = classifier(images)
            attack_loss = criterion(output_classifier, torch.arange(args.n_classes).to(args.device))
            discrim_loss = BCELoss(discrim, torch.zeros_like(discrim).to(args.device))
            loss = attack_loss + args.w_gan * discrim_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            Total_attack_loss.update(attack_loss.item(), inputs.size(0))
            Total_loss.update(loss.item(), inputs.size(0))
            Acc_total.update( output_classifier.argmax(1).eq(torch.arange(args.n_classes).to(args.device)).sum().item() / args.n_classes, inputs.size(0))
        pbar.set_description(
                f'Epoch: {epoch + 1}/{args.epochs} | '
                f'Loss_total: {Total_loss.avg:.4f} | '
                f'Loss_attack: {Total_attack_loss.avg:.4f} | '
                f'Acc_total: {Acc_total.avg:.4f} | '
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


def evaluate(wandb, args, classifier_val, Attacker, G):
    Attacker.eval()
    classifier_val.eval()
    G.eval()
    
    inputs = torch.eye(args.n_classes).to(args.device).float()
    outputs = Attacker(inputs)
    images = G(outputs)
    output_classifier = classifier_val(images)
    val_acc = output_classifier.argmax(1).eq(torch.arange(args.n_classes).to(args.device)).float().mean().item()

    image_array = rearrange(images, 
                'b c h w -> (b h) w c').cpu().detach().numpy().astype(np.float64)
    images = wandb.Image(image_array, caption=f'Acc is {val_acc*100:2.2f}')


    return images, val_acc 


def main():
    args = para_config()
    if args.wandb_active:
        wandb.init(project= args.wandb_project,
                   entity = args.wandb_id,
                   config = args,
                   name = f'Dataset: {args.target_dataset}',
                   group = f'Dataset: {args.target_dataset}')
    else:
        os.environ["WANDB_SILENT"] = "true"
    print(args)
    if not torch.cuda.is_available():
        raise ValueError("Should buy a GPU!")
    set_random_seeds(args.random_seed)

    # load target classifier
    if args.img_size == 32:
        from models.resnet_32x32 import resnet10 as resnet
        from models.resnet_32x32 import resnet50 as resnet_val
    else:
        from models.resnet import resnet18 as resnet
        from models.resnet import resnet50 as resnet_val

    classifier = resnet(
        num_classes=args.n_classes,
        num_channel=args.num_channel).to(args.device)
    classifier_val = resnet_val(
        num_classes=args.n_classes,
        num_channel=args.num_channel).to(args.device)

    if os.path.exists(args.ckpt_path):
        ckpt = load_ckpt(args.ckpt_path, is_best = True)
        classifier.load_state_dict(ckpt['model'])
        classifier.eval()
        print(f'{args.ckpt_path} model is loaded!')
    else:
        raise Exception('there is no generative checkpoint')

    if os.path.exists(args.ckpt_path + '_valid'):
        ckpt = load_ckpt(args.ckpt_path + '_valid', is_best = True)
        classifier_val.load_state_dict(ckpt['model'])
        classifier_val.eval()
        print(f'{args.ckpt_path}_valid model is loaded!')
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
        ckpt = load_ckpt(args.ckpt_path_gan, is_best = True)
        G.load_state_dict(ckpt['model_g'])
        G.eval()
        D.load_state_dict(ckpt['model_d'])
        D.eval()
        print(f'{args.ckpt_path_gan} model is loaded!')
    else:
        raise Exception('there is no generative checkpoint')

    train(wandb, args, classifier, classifier_val, G, D)



if __name__ == "__main__":
    main() 
