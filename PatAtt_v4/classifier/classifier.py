"""========Default import===="""
from __future__ import print_function
import argparse
from utils.base_utils import set_random_seeds, get_accuracy, AverageMeter, WarmupCosineSchedule, WarmupLinearSchedule, save_ckpt, load_ckpt, get_data_loader, accuracy, CutMix
import torch.nn as nn
import torch
import os
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from einops import rearrange
import wandb
import torch.nn.functional as F
from models.resnet_32x32 import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights, resnext101_32x8d, ResNeXt101_32X8D_Weights, resnext101_64x4d, ResNeXt101_64X4D_Weights
from models.dla import DLA
from models.VGG import VGG11, VGG16, VGG19
from models.resnext_32x32 import ResNeXt29_32x4d
from models.shuffle_v2 import ShuffleNetG2
from models.mobilnet import MobileNetV2
""" ==========END================"""




def parse_args():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    #----------Default Optimizer
    parser.add_argument("--pin-memory", 
            type=bool,
            default = True)
    parser.add_argument("--weight-decay", 
            type=float,
            default = 5e-4)
    parser.add_argument("--decay-type",
            type=str,
            default='cosine')
    parser.add_argument("--epochs",
            type=int,
            default=20)
    parser.add_argument("--lr",
            type=float,
            default= 1e-1)
    parser.add_argument("--val",
            action=argparse.BooleanOptionalAction,
            default=False)
    parser.add_argument("--eval",
            action=argparse.BooleanOptionalAction,
            default=False)
    parser.add_argument("--browse",
            action=argparse.BooleanOptionalAction,
            default=False)
    parser.add_argument("--cutmix",
            action=argparse.BooleanOptionalAction,
            default=False)
    parser.add_argument("--warmup-steps",
            type=int,
            default = 100)
    #----------Dataset Loader
    parser.add_argument("--dataset", 
            type=str,
            default = 'mnist')
    parser.add_argument("--num-workers", 
            type=int,
            default = 4)
    parser.add_argument("--train-batch-size", 
            type=int,
            default = 256)
    parser.add_argument("--test-batch-size", 
            type=int,
            default = 256)
    #------------configurations
    parser.add_argument("--ckpt-fpath",
            type=str,
            default='../experiments/classifier/',
            help="save path.")
    parser.add_argument("--interval-val",
            type=int,
            default = 1)
    #--------------wandb
    parser.add_argument("--wandb-project",
            type=str,
            default='classifiers')
    parser.add_argument("--wandb-id",
            type=str,
            default= 'jonggyujang0123')
    parser.add_argument("--wandb-name",
            type=str,
            default= 'classifier')
    parser.add_argument("--wandb-active",
            type=bool,
            default=True)
    args = parser.parse_args()
    #-----------------set image configurations
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.dataset == 'mnist' or args.dataset == 'HAN':
        args.num_channel = 1
        args.img_size = 32
        args.num_classes = 10
    if args.dataset == 'emnist':
        args.num_channel = 1
        args.img_size = 32
        args.num_classes = 26
    if args.dataset in ['cifar10','caltech101', 'cifar100']:
        args.num_channel = 3
        args.img_size = 32
        args.num_classes = 10
    if args.dataset in 'celeba' or args.dataset in 'LFW':
        args.num_channel = 3
        args.img_size = 128
        args.num_classes = 300
    return args



def train(wandb, args, model):
    ## Setting 
    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    train_loader, test_loader, _ = get_data_loader(args=args)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs*0.5), int(args.epochs*0.75), int(args.epochs*0.9)], gamma=0.1)
    val_acc = .0; best_acc = .0; start_epoch =0
    AvgLoss = AverageMeter()
    # We only save the model who uses device "cuda:0"

    cutmix = CutMix(alpha=1.0)
    #  cutmix = Mixup(alpha=1.0)
    if args.cutmix:
        cutmix_prob = 0.5
    else:
        cutmix_prob = 0.0
    
    # Prepare dataset and dataloader
    for epoch in range(start_epoch, args.epochs):
#        if args.local_rank not in [-1,1]:
#            print("Local Rank: {}, Epoch: {}, Training ...".format(args.local_rank, epoch))

        model.train()
        tepoch = tqdm(train_loader)
        for batch, data in enumerate(tepoch):
            optimizer.zero_grad()
            inputs, labels = data[0].to(args.device), data[1].to(args.device)
            if args.cutmix and np.random.rand() < cutmix_prob:
                inputs, labela, labelb, lam = cutmix(inputs, labels)
                output = model(inputs)
                loss = criterion(output, labela) * lam + criterion(output, labelb) * (1. - lam)
            else:
                outputs = model(inputs)
                loss =criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            AvgLoss.update(loss.item(), n =len(data))
            tepoch.set_description(f'Epoch {epoch}: train_loss: {AvgLoss.avg:2.2f}, lr : {scheduler.get_lr()[0]:.2E}')

        scheduler.step()
        # save and evaluate model routinely.
        if epoch % args.interval_val == 0:
            val_acc, top5_acc = evaluate(args, model=model, device=args.device, val_loader=test_loader)
            ckpt = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch
                    }
            save_ckpt(checkpoint_fpath=args.ckpt_fpath, checkpoint=ckpt)
            if best_acc < val_acc:
                save_ckpt(checkpoint_fpath=args.ckpt_fpath, checkpoint=ckpt, is_best=True)
                best_acc = val_acc
            if args.wandb_active:
                wandb.log({
                    "loss": AvgLoss.val,
                    "top1" : val_acc,
                    "top5" : top5_acc,
                    })
            print("-"*75+ "\n")
            print(f"| {epoch}-th epoch, training loss is {AvgLoss.avg}, and Val Accuracy is {val_acc*100:2.2f}, top5 accuracy is {top5_acc*100:2.2f}\n")
            print("-"*75+ "\n")
        AvgLoss.reset()
        tepoch.close()


def evaluate(args, model, val_loader, device):
    model.eval()
    acc_t1 = AverageMeter()
    acc_t5 = AverageMeter()
    valepoch = tqdm(val_loader, 
                  unit= " batch")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(valepoch):
            inputs = inputs.to(device, non_blocking=True)
            # compute the output
            output = model(inputs).detach().cpu()

            # Measure accuracy
            top1, top5 = accuracy(output, labels, topk=(1, 5))
            acc_t1.update(top1.item(), n=inputs.size(0))
            acc_t5.update(top5.item(), n=inputs.size(0))
            valepoch.set_description(f'Validation Acc: {acc_t1.avg:2.4f} | {acc_t5.avg:2.4f}')
            #  acc.update(acc_batch, n=inputs.size(0))
            #  valepoch.set_description(f'Validation Acc: {acc.avg:2.2f}')
    valepoch.close()
    #  result  = torch.tensor([acc.sum,acc.count]).to(device)

    return acc_t1.avg, acc_t5.avg


def browse(wandb, args):
    import torchvision
    from torchvision.utils import save_image
    train_loader, _, _ = get_data_loader(args=args, class_wise=True)
    images = []
    n_row = 10
    for classes in range(10):
        print(f'{classes}th class')
        loader = train_loader[classes]
        for k, (image, idx) in enumerate(loader):
            if k > 0:
                continue
            images.append(image[0:n_row, :, :, :])
            dir_name = f'./Results/GT_images/{args.dataset}/{idx[0]}'
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            for i in range(image.size(0)):
                tensor = (image[i, ...].cpu().detach() + 1)/2
                if args.num_channel == 1:
                    tensor = torch.cat([tensor, tensor, tensor], dim = 0)
                save_image(tensor, f'{dir_name}/{i}.png')

    images = torch.cat(images, dim=0)
    images = F.pad(images, pad = (1, 1, 1, 1), value=-1)
    images = rearrange(images, '(b1 b2) c h w -> (b2 h) (b1 w) c', b1=10, b2=n_row).numpy().astype(np.float64)
    images = wandb.Image(images)
    wandb.log({"{args.dataset}": images})
    

    


def main():
    args = parse_args()
    if args.val == True:
        args.ckpt_fpath = f"{args.ckpt_fpath}/{args.dataset}_valid"
    else:
        args.ckpt_fpath = f"{args.ckpt_fpath}/{args.dataset}"

    if args.wandb_active:
        wandb.init(project = args.wandb_project,
                   entity = args.wandb_id,
                   config = args,
                   name = f'{args.dataset}',
                   group = args.dataset
                   )
    print(args)
    # set cuda flag
    if not torch.cuda.is_available():
        print("WARNING: You have a CUDA device, so you should probably enable CUDA")
    # Set Automatic Mixed Precision
    # We need to use seeds to make sure that model initialization same
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    args.device = torch.device("cuda:0")
   
    if args.img_size == 32:
        if args.val:
            model = DLA(num_channel=args.num_channel, num_classes=args.num_classes)
            set_random_seeds(random_seed = 0)
        else:
            #  model = ResNet18(num_channel=args.num_channel, num_classes=args.num_classes)
            #  model = ResNet50(num_channel=args.num_channel, num_classes=args.num_classes)
            #  model = ResNeXt29_32x4d(num_channel=args.num_channel, num_classes=args.num_classes)
            #  model = VGG19(num_channel=args.num_channel, num_classes=args.num_classes)
            model = MobileNetV2()

            set_random_seeds(random_seed = 7)
    else:
        if args.val:
            model = resnext50_32x4d(weights = ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
            model.fc = nn.Linear(2048, args.num_classes)
            model = nn.Sequential(
                    nn.Upsample(scale_factor=2.0, mode='bilinear'),
                    model
                    )
            set_random_seeds(random_seed = 0)
        else:
            model = resnext50_32x4d(weights = ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
            model.fc = nn.Linear(2048, args.num_classes)
            set_random_seeds(random_seed = 7)


    model = model.to(args.device)
    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable parameter is {pytorch_total_params:.2E}')

    if args.browse:
        browse(wandb, args)
    elif args.eval:
        train_loader, val_loader, _ = get_data_loader(args=args)
        val_acc, top5_acc = evaluate(args, model, val_loader, args.device)
        print(f'Validation Acc: {val_acc:2.4f} | {top5_acc:2.4f}')
    else:
        train(wandb, args, model)


if __name__ == '__main__':
    main()
