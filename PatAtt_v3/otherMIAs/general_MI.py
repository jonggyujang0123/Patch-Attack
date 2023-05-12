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
"""import target classifier, generator, and discriminator """


import argparse

def para_config():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    # hyperparameter setting
    parser.add_argument("--epochs",
            type=int,
            default=200)
    parser.add_argument("--random-seed",
            type=int,
            default=0)
    parser.add_argument("--eval-every",
            type=int,
            default=5)
    parser.add_argument("--pin-memory",
            type=bool,
            default=True)

    # dataset 
    parser.add_argument("--n-classes",
            type=int,
            default=10)
    parser.add_argument("--img-size",
            type=int,
            default=32)
    parser.add_argument("--num-workers",
            type=int,
            default=4)
    parser.add_argument("--num-channel",
            type=int,
            default=1)
    
    # save path configuration
    parser.add_argument("--target-dataset",
            type=str,
            default="mnist",
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


    # optimizer setting 
    parser.add_argument("--weight-decay",
            type=float,
            default = 5.0e-4)
    parser.add_argument("--beta-1",
            type=float,
            default = 0.5)
    parser.add_argument("--beta-2",
            type=float,
            default = 0.999)
    parser.add_argument("--decay-type",
            type=str,
            default="linear",
            help="choose linear or cosine")
    parser.add_argument("--warmup-steps",
            type=int,
            default=100)
    parser.add_argument("--lr",
            type=float,
            default=1e-4,
            help = "learning rate")
    

    args = parser.parse_args()
    return args


def train(wandb, args, classifier, classifier_val):
    Attacker = nn.Sequential(
            Rearrange('b c -> b c () ()'),
            nn.ConvTranspose2d(args.num_channel, args.num_channel, args.img_size, 1)
            )
    CE_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(Attacker.parameters(), lr =args.lr, betas = (args.beta_1, args.beta_2))
    if args.decay_type == "cosine":
        scheduler= WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.epochs*100)
    else:
        scheduler= WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.epochs*100)
    start_epoch = 0
    # We only save the model who uses device "cuda:0"
    # To resume the device for the save model woule be also "cuda:0"
    Loss_attack = AverageMeter()
    Acc_total_t = AverageMeter()

    pbar = tqdm(
            range(start_epoch,args.epochs),
            disable = args.local_rank not in [-1,0],
            bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
            )
#    pad = Pad(args.patch_size - args.patch_stride, fill=0)
    for epoch in pbar:
    # Prepare dataset and dataloader
        # Switch Training Mode
        tepoch = tqdm(
                range(100),
                bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
                )
        for data in range(100):
            inputs = torch.eye(args.n_classes).to(args.device)
            optimizer.zero_grad()
            fake = G(inputs)
            D_disc = classifier(fake)
            loss_attack = CE_loss(D_disc, inputs.max(1)[1])
            loss_attack.backward()
            optimizer.step()
            scheduler.step()
            Loss_attack.update(loss_attack.detach().item(), inputs.size(0))
            train_Acc = D_disc.softmax(1)[torch.arange(inputs.size(0)),inputs.max(1)[1]].mean().item()
            Acc_total_t.update(train_Acc, inputs.size(0))

            tepoch.set_description(f'Ep {epoch}: L_D : {Loss_d.avg:2.3f}, L_G: {Loss_g.avg:2.3f}, L_info: {Loss_info.avg:2.3f}, lr: {scheduler_d.get_lr()[0]:.1E}, Acc:{Acc_total_t.avg:2.3f}, Max_A: {Max_act.avg:2.1f}')
        if epoch % args.eval_every == args.eval_every - 1 or epoch==0:
            _, images = evaluate(wandb, args, classifier_val, G, epoch = epoch)
            if args.wandb_active:
                wandb.log({
                        "val_loss" : Loss_attack.avg,
                        "val_acc_t" : Acc_total_t.avg,
                        "image" : images,
                        },
                        step = epoch)
        Acc_total_t.reset()
        Loss_attack.reset()


def evaluate(wandb, args, classifier, G, fixed_z=None, fixed_c = None, epoch = 0):
    G.eval()
    if fixed_z == None:
        fixed_z = torch.randn(
                args.test_batch_size*args.x_sample, 
                args.latent_size,
                ).to(args.device)
    if fixed_c == None:
        fixed_c = torch.multinomial(torch.ones(args.n_classes), args.test_batch_size*args.x_sample, replacement=True).to(args.device)
    val_acc = 0
    if args.local_rank in [-1,0]:
        fake = G(fixed_z, fixed_c)
        pred = classifier( fake)
        fake_y = fixed_c
#        fake_y = args.fixed_id * \
#                torch.ones(cfg.test_batch_size).to(cfg.device)
        val_acc = (pred.max(1)[1] == fake_y).float().mean().item()
        pred_sort =  pred[range(fixed_c.shape[0]), fixed_c].reshape(args.n_classes, args.test_batch_size * args.x_sample // args.n_classes).topk(args.test_batch_size // args.n_classes, dim=-1)[1]
        fake = fake.reshape(args.n_classes, args.test_batch_size*args.x_sample//args.n_classes, args.num_channel, args.img_size, args.img_size)[torch.arange(args.n_classes), pred_sort, :, :, :]
        image_array = rearrange(fake, 
                'b1 b2 c h w -> (b2 h) (b1 w) c').cpu().detach().numpy().astype(np.float64)
        images = wandb.Image(image_array, caption=f'Acc is {val_acc*100:2.2f}')
    return val_acc, images



def main():
    args = para_config()
    args.ckpt_fpath = os.path.join(args.ckpt_dir, args.dataset)
    
    if args.wandb_active:
        wandb.init(project = args.wandb_project,
                   entity = args.wandb_id,
                   config = args,
                   name = f'Dataset: {args.dataset}',
                   group = f'Dataset: {args.dataset}'
                   )
    print(args)
    # set cuda flag
    if not torch.cuda.is_available():
        print("WARNING: You have a CUDA device, so you should probably enable CUDA")
    # Set Automatic Mixed Precision
    # We need to use seeds to make sure that model initialization same
    set_random_seeds(random_seed = args.random_seed)
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    args.device = torch.device("cuda:0")

    # torch.distributed.init_process_group(backend="gloo")

    # Encapsulate the model on the GPU assigned to the current process
    if args.img_size == 32:
        from models.resnet_32x32 import resnet10 as resnet
        from models.resnet_32x32 import resnet50 as resnet_val
    else:
        from models.resnet import resnet18 as resnet
        from models.resnet import resnet50 as resnet_val

        
    classifier = resnet(
            num_classes= args.n_classes,
            num_channel= args.num_channel,
            ).to(args.device)
    classifier_val = resnet_val(
            num_classes= args.n_classes,
            num_channel= args.num_channel,
            ).to(args.device)


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
    else:
        raise Exception('there is no generative checkpoint')
    # Load target classifier 

    train(wandb, args, classifier, classifier_val)
    wandb.finish()


if __name__ == '__main__':
    main()
