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
"""import target classifier, generator, and discriminator """

import argparse

def para_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',
                        type=int,
                        default=50)
    parser.add_argument('--train-batch-size',
                        type=int,
                        default=64)
    parser.add_argument('--test-batch-size',
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
    parser.add_argument('--latent-size',
                        type=int,
                        default=64)
    parser.add_argument('--w-gan',
                        type=float,
                        default=0.15)
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
    parser.add_argument('--num-workers',
                        type=int,
                        default=4)
    parser.add_argument('--pin-memory',
                        type=bool,
                        default=True)
    parser.add_argument('--ckpt-dir',
                        type=str,
                        default='../experiments')
    parser.add_argument("--max-classes",
            type=int,
            default=10,
            help="maximum number of classes")


    args = parser.parse_args()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.ckpt_path = os.path.join(args.ckpt_dir, 'classifier', args.target_dataset)
    args.ckpt_path_gan = os.path.join(args.ckpt_dir, 'common_gan', args.aux_dataset)
    if args.target_dataset in ['mnist', 'fashionmnist', 'kmnist']:
        args.num_channel = 1
        args.img_size = 32
        args.num_classes = 10
    if args.target_dataset == 'emnist':
        args.num_channel = 1
        args.img_size = 32
        args.num_classes = 26
    if args.target_dataset == 'cifar10':
        args.num_channel = 3
        args.img_size = 32
        args.num_classes = 10
    if args.target_dataset == 'cifar100':
        args.num_channel = 3
        args.img_size = 32
        args.num_classes = 100
    if args.target_dataset == 'LFW':
        args.num_channel = 3
        args.img_size = 64
        args.num_classes = 10
    if args.target_dataset == 'celeba':
        args.num_channel = 3
        args.img_size = 64
        args.num_classes = 1000
    return args



def train(wandb,  args, classifier, classifier_val, G, D):
    Attacker = nn.Sequential(
            nn.Linear(min(args.num_classes, args.max_classes), args.latent_size),
            nn.ReLU(),
            nn.Linear(args.latent_size, args.latent_size),
            nn.ReLU(),
            nn.Linear(args.latent_size, args.latent_size),
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
            inputs = torch.eye(min(args.num_classes, args.max_classes)).to(args.device).float()
            optimizer.zero_grad()
            outputs = Attacker(inputs)
            images = G(outputs)
            discrim = D(images)
            output_classifier = classifier(images)
            attack_loss = criterion(output_classifier, 
                                    torch.arange(min(args.num_classes, args.max_classes)).to(args.device))
            discrim_loss = BCELoss(discrim, torch.zeros_like(discrim).to(args.device))
            loss = attack_loss + args.w_gan * discrim_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            Total_attack_loss.update(attack_loss.item(), inputs.size(0))
            Total_loss.update(loss.item(), inputs.size(0))
            Acc_total.update( output_classifier.argmax(1).eq(torch.arange(
                min(args.num_classes, args.max_classes)
                ).to(args.device)).sum().item() / args.num_classes, inputs.size(0))
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
    test(wandb, args, classifier_val, Attacker, G)

def evaluate(wandb, args, classifier_val, Attacker, G):
    Attacker.eval()
    classifier_val.eval()
    G.eval()
    
    inputs = torch.eye(
            min(args.num_classes, args.max_classes)
            ).to(args.device).float()
    outputs = Attacker(inputs)
    images = G(outputs)
    output_classifier = classifier_val(images)
    val_acc = output_classifier.argmax(1).eq(torch.arange(min(args.num_classes, args.max_classes)).to(args.device)).float().mean().item()
    fake = F.pad(images, pad = (1,1,1,1), value = -1)
    
    image_array = rearrange(images, 
                'b c h w -> h (b w) c').cpu().detach().numpy().astype(np.float64)
    images = wandb.Image(image_array, caption=f'Acc is {val_acc*100:2.2f}')


    return images, val_acc 



def test(wandb, args, classifier_val,  Attacker, G):
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics import StructuralSimilarityIndexMeasure
    #  from sentence_transformers import SentenceTransformer, util
    #  model = SentenceTransformer('clip-ViT-B-32', device = args.device)
    #  import torchvision.transforms as T
    #  from PIL import Image
    #  transform = T.ToPILImage()
    Attacker.eval()
    Acc_val = AverageMeter() 
    Acc_val_total = []
    Top5_val = AverageMeter()
    Top5_val_total = []
    Conf_val = AverageMeter()
    Conf_val_total = []
    SSIM_val = AverageMeter()
    SSIM_val_total = []
    #  CLIP_val = AverageMeter()
    #  CLIP_val_total = []
    FID_val = AverageMeter()
    FID_val_total = []
    target_dataset, _, _ = get_data_loader(args, args.target_dataset, class_wise = True)
    for class_ind in range(min(args.num_classes, args.max_classes)):
        dataset = target_dataset[class_ind]
        pbar = tqdm(enumerate(dataset), total=len(dataset))
        fid = FrechetInceptionDistance(feature=64, compute_on_cpu = True).to(args.device)
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0, channel = args.num_channel, compute_on_gpu=True).to(args.device)
        for batch_idx, (x, y) in pbar:
            label = y.to(args.device)
            inputs = torch.zeros(label.size(0), min(args.num_classes, args.max_classes)).to(args.device).float()
            inputs[:,label[0]] = 1
            fake = G(Attacker(inputs))
            fake = fake.detach()
            x = x.to(args.device)
            pred_val = classifier_val(fake)
            pred_val = pred_val.detach()
            x_int  = (255 * (x+1)/2).type(torch.uint8)
            fake_int  = (255 * (fake+1)/2).type(torch.uint8)
            if args.num_channel == 1:
                x_int = x_int.repeat(1,3,1,1)
                fake_int = fake_int.repeat(1,3,1,1)
            fid.update(x_int, real = True)
            fid.update(fake_int, real = False)
            top1, top5 = accuracy(pred_val, label, topk=(1, 5))
            Acc_val.update(top1.item(), x.shape[0])
            Top5_val.update(top5.item(), x.shape[0])
            sftm = pred_val.softmax(dim=1)
            Conf_val.update(sftm[:, class_ind].mean().item(), x.shape[0] )
            #  clip_list = []
            #  for i in range(x.shape[0]):
            #      lists = [transform((x[i,:,:,:]+1)/2), transform((fake[i,:,:,:]+1)/2)]
            #      encoded = model.encode(lists, batch_size=128, convert_to_tensor=True)
            #      score = util.paraphrase_mining_embeddings(encoded)
            #      clip_list.append(score[0][0])
            #  CLIP_val.update(np.mean(clip_list), x.shape[0])
            ssim_list = []
            for i in range(x.shape[0]):
                ssim_score =ssim( (x[i:i+1,:,:,:]+1)/2, (fake[i:i+1,:,:,:]+1)/2)
                ssim_list.append(ssim_score.item())
            SSIM_val.update(np.mean(ssim_list), x.shape[0])
        fid_score = fid.compute()
        fid.reset()
        Acc_val_total.append(Acc_val.avg) 
        Conf_val_total.append(Conf_val.avg)
        SSIM_val_total.append(SSIM_val.avg)
        #  CLIP_val_total.append(CLIP_val.avg)
        FID_val_total.append(fid_score.item())
        Top5_val_total.append(Top5_val.avg)

        print(
            f'==> Testing model.. target class: {class_ind}\n'
            f'    Acc: {Acc_val.avg}\n'
            f'    Top5: {Top5_val.avg}\n'
            f'    Conf: {Conf_val.avg}\n'
            f'    SSIM score : {SSIM_val.avg}\n'
            #  f'    CLIP score : {CLIP_val.avg}\n'
            f'    FID score : {fid_score}\n'
            )
        Acc_val.reset()
        Conf_val.reset()
        #  CLIP_val.reset()
        SSIM_val.reset()
        FID_val.reset()
        Top5_val.reset()
    print(
        f'==> Overall Results\n'
        f'    Acc: {np.mean(Acc_val_total)}\n'
        f'    Conf: {np.mean(Conf_val_total)}\n'
        #  f'    CLIP score : {np.mean(CLIP_val_total)}\n'
        f'    SSIM score : {np.mean(SSIM_val_total)}\n'
        f'    FID score : {np.mean(FID_val_total)}\n'
        )


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
        from models.resnet import resnet34 as resnet
        from models.resnet import resnet50 as resnet_val

    classifier = resnet(
        num_classes=args.num_classes,
        num_channel=args.num_channel).to(args.device)
    classifier_val = resnet_val(
        num_classes=args.num_classes,
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
