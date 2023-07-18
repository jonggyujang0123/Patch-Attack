"""========Default import===="""
from __future__ import print_function
import argparse
from utils.base_utils import set_random_seeds, get_accuracy, AverageMeter, WarmupCosineSchedule, WarmupLinearSchedule, save_ckpt, load_ckpt, get_data_loader, accuracy
import torch.nn as nn
import torch
import os
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from einops import rearrange
import wandb
import torch.nn.functional as F 
""" ==========END================"""




def parse_args():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    #----------Default Optimizer
    parser.add_argument("--resume",
            type=bool,
            default= False,
            help="Resume the last training from saved checkpoint.")
    parser.add_argument("--test", 
            type=bool,
            default = False,
            help="if test, choose True.")
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
    parser.add_argument("--valid",
            type=bool,
            default = False)
    parser.add_argument("--warmup-steps",
            type=int,
            default = 100)
    #----------Dataset Loader
    parser.add_argument("--dataset", 
            type=str,
            default = 'mnist')
    parser.add_argument("--num-workers", 
            type=int,
            default = 8)
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
    if args.dataset in ['mnist', 'fashionmnist', 'kmnist']:
        args.num_channel = 1
        args.img_size = 32
        args.num_classes = 10
    if args.dataset == 'emnist':
        args.num_channel = 1
        args.img_size = 32
        args.num_classes = 26
    if args.dataset == 'cifar10':
        args.num_channel = 3
        args.img_size = 32
        args.num_classes = 10
    if args.dataset == 'cifar100':
        args.num_channel = 3
        args.img_size = 32
        args.num_classes = 100
    if args.dataset == 'LFW':
        args.num_channel = 3
        args.img_size = 64
        args.num_classes = 10
    if args.dataset == 'celeba':
        args.num_channel = 3
        args.img_size = 64
        args.num_classes = 1000
    return args

def train(wandb, args, model):
    ## Setting 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    train_loader, test_loader, _ = get_data_loader(args=args)
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.epochs*len(train_loader))
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.epochs*len(train_loader))
    val_acc = .0; best_acc = .0; start_epoch =0
    AvgLoss = AverageMeter()
    # We only save the model who uses device "cuda:0"
    # To resume the device for the save model woule be also "cuda:0"
    if args.resume:
        ckpt = load_ckpt(args.ckpt_fpath)
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']+1


    # Prepare dataset and dataloader
    for epoch in range(start_epoch, args.epochs):
#        if args.local_rank not in [-1,1]:
#            print("Local Rank: {}, Epoch: {}, Training ...".format(args.local_rank, epoch))

        model.train()
        tepoch = tqdm(train_loader)
        for batch, data in enumerate(tepoch):
            optimizer.zero_grad()
            inputs, labels = data[0].to(args.device), data[1].to(args.device)
            outputs = model(inputs)
            loss =criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            AvgLoss.update(loss.item(), n =len(data))
            tepoch.set_description(f'Epoch {epoch}: train_loss: {AvgLoss.avg:2.2f}, lr : {scheduler.get_lr()[0]:.2E}')

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

def test(wandb, args):
    train_loader, _, _ = get_data_loader(args=args, class_wise=True)
    images = []
    for classes in range(10):
        print(f'{classes}th class')
        loader = train_loader[classes]
        for k, (image, idx) in enumerate(loader):
            if k > 0:
                continue
            images.append(image[0:4, :, :, :])

    images = torch.cat(images, dim=0)
    images = F.pad(images, pad = (1, 1, 1, 1), value=-1)
    images = rearrange(images, '(b1 b2) c h w -> (b2 h) (b1 w) c', b1=10, b2=4).numpy().astype(np.float64)
    images = wandb.Image(images)
    wandb.log({"{args.dataset}": images})
    

    


def main():
    args = parse_args()
    if args.valid == True:
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
        if args.valid:
            #  from models.resnet_32x32 import resnet50 as resnet
            from models.resnet import resnet50 as resnet
            set_random_seeds(random_seed = 7)
        else:
            #  from models.resnet_32x32 import resnet10 as resnet
            from models.resnet import resnet10 as resnet
            set_random_seeds(random_seed = 0)
    else:
        if args.valid:
            from models.resnet import resnet50 as resnet
            set_random_seeds(random_seed = 7)
        else:
            from models.resnet import resnet34 as resnet
            set_random_seeds(random_seed = 0)


    # torch.distributed.init_process_group(backend="gloo")

    # Encapsulate the model on the GPU assigned to the current process
    model = resnet(num_classes=args.num_classes, num_channel= args.num_channel, pretrained = True if args.img_size == 64 and args.valid else False)
    # if custom_pre-trained model : model.load_from(np.load(<path>))
    model = model.to(args.device)
    if args.resume :
        ckpt = load_ckpt(args.ckpt_fpath, is_best = args.test)
        model.load_state_dict(ckpt['model'])

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable parameter is {pytorch_total_params:.2E}')

    if args.test:
        test(wandb, args)
    else:
        train(wandb, args, model)


if __name__ == '__main__':
    main()
