"""Import default libraries"""
import os
import argparse
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
from torch.nn import functional as F
from models.resnet_32x32 import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
"""import target classifier, generator, and discriminator """



def para_config():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    # hyperparameter setting
    parser.add_argument("--epochs",
            type=int,
            default=10)
    parser.add_argument("--random-seed",
            type=int,
            default=0)
    parser.add_argument("--pin-memory",
            type=bool,
            default=True)
    parser.add_argument("--n-images",
            type=int,
            default=1000)
    parser.add_argument("--device",
            type=str,
            default="cuda:0")


    # dataset 
    parser.add_argument("--num-workers",
            type=int,
            default=4)
    
    # save path configuration
    parser.add_argument("--target-dataset",
            type=str,
            default="emnist",
            help="choose the target dataset")
    parser.add_argument("--ckpt-dir",
            type=str,
            default="../experiments/classifier/",
            help="path for restoring the classifier")

    # WANDB SETTINGS
    parser.add_argument("--wandb-project",
            type=str,
            default="General-MI")
    parser.add_argument("--wandb-id",
            type=str,
            default="jonggyujang0123")
    parser.add_argument("--wandb-name",
            type=str,
            default="generalMI")
    parser.add_argument("--wandb-active",
            type=bool,
            default=True)
    parser.add_argument("--target-class",
            type=int,
            default=0)

    # optimizer setting 
    parser.add_argument("--weight-decay",
            type=float,
            default = 5.0e-4)
    parser.add_argument("--beta-1",
            type=float,
            default = 0.1)
    parser.add_argument("--beta-2",
            type=float,
            default = 0.1)
    parser.add_argument("--decay-type",
            type=str,
            default="linear",
            help="choose linear or cosine")
    parser.add_argument("--warmup-steps",
            type=int,
            default=100)
    parser.add_argument("--lr",
            type=float,
            default=3e-1,
            help = "learning rate")
    args = parser.parse_args()
    args.ckpt_path = os.path.join(args.ckpt_dir, args.target_dataset)
    if args.target_dataset in ['mnist']:
        args.num_channel = 1
        args.img_size = 32
        args.num_classes = 10
    elif args.target_dataset in ['emnist']:
        args.num_channel = 1
        args.img_size = 32
        args.num_classes = 26
    elif args.target_dataset in ['cifar10']:
        args.num_channel = 3
        args.img_size = 32
        args.num_classes = 10
    else:
        raise ValueError("Please choose the dataset in ['mnist', 'emnist', 'cifar10']")
    return args


def train(wandb, args, classifier, classifier_val):
    Attacker = torch.autograd.Variable(
            torch.randn(args.n_images, 
                        args.num_channel, 
                        args.img_size, 
                        args.img_size).to(args.device),
            requires_grad=True)
    optimizer = optim.Adam([Attacker], lr =args.lr, betas = (args.beta_1, args.beta_2))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5*args.epochs, 0.75*args.epochs], gamma=0.1)
    Loss_attack = AverageMeter()
    Acc_total_t = AverageMeter()

    pbar = tqdm(
            range(args.epochs),
            bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
            )
    for epoch in pbar:
        for _ in range(100):
            optimizer.zero_grad()
            logits = classifier(Attacker.tanh())
            loss_attack = - logits.softmax(1)[:,args.target_class].log().mean()
            loss_attack.backward()
            optimizer.step()

            train_acc = logits.argmax(1).eq(args.target_class).float().mean().item()
            Loss_attack.update(loss_attack.detach().item(), 1)
            Acc_total_t.update(train_acc, 1)
        pbar.set_description(f'Ep {epoch}: L_attack : {Loss_attack.avg:2.3f}, lr: {scheduler.get_lr()[0]:.1E}, Acc:{Acc_total_t.avg:2.3f}')
        scheduler.step()

        val_acc, images = evaluate(wandb, args, classifier_val, Attacker, epoch = epoch)
        if args.wandb_active:
            wandb.log({
                    "val_loss" : Loss_attack.avg,
                    "acc_train" : Acc_total_t.avg,
                    "val_acc" : val_acc,
                    "image" : images,
                    },
                    step = epoch)
        Acc_total_t.reset()
        Loss_attack.reset()
    return Attacker

def evaluate(wandb, args, classifier, Attacker, epoch = 0):
    pred = classifier(Attacker.data.tanh())
    val_acc = pred.argmax(1).eq(args.target_class).float().mean().item()
    if args.num_channel == 1:
        fake = F.pad(Attacker.data.tanh(), pad = (1,1,1,1), value = 1)
    else:
        fake = F.pad(Attacker.data.tanh(), pad = (1,1,1,1), value = -1)
    image_array = rearrange(fake[:100, ...], '(b1 b2) c h w -> (b1 h) (b2 w) c', b1 = 10, b2 = 10).cpu().detach().numpy().astype(np.float64)
    images = wandb.Image(image_array, caption=f'Acc is {val_acc*100:2.2f}')
    return val_acc, images

def save_images(args, Attacker):
    from torchvision.utils import save_image
    import torchvision

    fake = Attacker.data.tanh()
    directory = f'./Results/General_MI/{args.target_dataset}/{args.target_class}'
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
        wandb.init(project = args.wandb_project,
                   entity = args.wandb_id,
                   config = args,
                   name = f'Dataset: {args.target_dataset}_{args.target_class}',
                   group = f'Dataset: {args.target_dataset}'
                   )
    else:
        os.environ['WANDB_MODE'] = 'dryrun'
    print(args)
    set_random_seeds(random_seed = args.random_seed)


    # Setup Models

    classifier = ResNet18(num_classes = args.num_classes, num_channel = args.num_channel).to(args.device)
    classifier_val = ResNet34(num_classes = args.num_classes, num_channel = args.num_channel).to(args.device)

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
    else:
        raise Exception('there is no generative checkpoint')

    Attacker = train(wandb, args, classifier, classifier_val)
    save_images(args, Attacker)

if __name__ == '__main__':
    main()
