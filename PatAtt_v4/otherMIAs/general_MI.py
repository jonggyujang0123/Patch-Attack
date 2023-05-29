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
"""import target classifier, generator, and discriminator """



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
            default=20)
    parser.add_argument("--pin-memory",
            type=bool,
            default=True)
    parser.add_argument("--test-batch-size",
            type=int,
            default=100)
    parser.add_argument("--local_rank",
            type=int,
            default=-1)
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
            default=3e-1,
            help = "learning rate")
    parser.add_argument("--max-classes",
            type=int,
            default=10,
            help="maximum number of classes")
    args = parser.parse_args()
    args.ckpt_path = os.path.join(args.ckpt_dir, args.target_dataset)
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


def train(wandb, args, classifier, classifier_val):
    Attacker = nn.Sequential(
            Rearrange('b c -> b c () ()'),
            nn.ConvTranspose2d(min(args.num_classes, args.max_classes), args.num_channel, args.img_size, 1, 0, bias=False),
            nn.Tanh()
            ).to(args.device)
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
    #  pad = Pad(args.patch_size - args.patch_stride, fill=0)
    for epoch in pbar:
    # Prepare dataset and dataloader
        # Switch Training Mode
        #  tepoch = tqdm(
        #          range(100),
        #          bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
        #          )
        for _ in range(100):
            inputs = torch.eye(min(args.num_classes, args.max_classes)).to(args.device).float()
            optimizer.zero_grad()
            fake = Attacker(inputs)
            D_disc = classifier(fake)
            loss_attack = CE_loss(D_disc, inputs.max(1)[1])
            loss_attack.backward()
            optimizer.step()
            scheduler.step()
            Loss_attack.update(loss_attack.detach().item(), inputs.size(0))
            train_Acc = D_disc.softmax(1)[torch.arange(inputs.size(0)),inputs.max(1)[1]].mean().item()
            Acc_total_t.update(train_Acc, inputs.size(0))
        pbar.set_description(f'Ep {epoch}: L_attack : {Loss_attack.avg:2.3f}, lr: {scheduler.get_lr()[0]:.1E}, Acc:{Acc_total_t.avg:2.3f}')

        if epoch % args.eval_every == args.eval_every - 1 or epoch==0:
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
    test(wandb, args, classifier_val, Attacker)




def evaluate(wandb, args, classifier, Attacker, epoch = 0):
    Attacker.eval()
    inputs = torch.eye(min(args.num_classes, args.max_classes)).to(args.device).float()
    fake = Attacker(inputs)
    pred = classifier(fake)
    fake_y = inputs.max(1)[1]
    val_acc = (pred.max(1)[1] == fake_y).float().mean().item()
    fake = F.pad(fake, pad = (1,1,1,1), value = -1)
    image_array = rearrange(fake, 
                'b c h w -> h (b w) c').cpu().detach().numpy().astype(np.float64)
    images = wandb.Image(image_array, caption=f'Acc is {val_acc*100:2.2f}')
    return val_acc, images

def test(wandb, args, classifier_val,  Attacker):
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
            fake = Attacker(inputs)
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
        wandb.init(project = args.wandb_project,
                   entity = args.wandb_id,
                   config = args,
                   name = f'Dataset: {args.target_dataset}',
                   group = f'Dataset: {args.target_dataset}'
                   )
    else:
        os.environ["WANDB_SILENT"] = "true"
    print(args)
    # set cuda flag
    if not torch.cuda.is_available():
        print("WARNING: You have a CUDA device, so you should probably enable CUDA")
    # Set Automatic Mixed Precision
    # We need to use seeds to make sure that model initialization same
    set_random_seeds(random_seed = args.random_seed)

    # Encapsulate the model on the GPU assigned to the current process
    if args.img_size == 32:
        from models.resnet_32x32 import resnet10 as resnet
        from models.resnet_32x32 import resnet50 as resnet_val
    else:
        from models.resnet import resnet34 as resnet
        from models.resnet import resnet50 as resnet_val

        
    classifier = resnet(
            num_classes= args.num_classes,
            num_channel= args.num_channel,
            ).to(args.device)
    classifier_val = resnet_val(
            num_classes= args.num_classes,
            num_channel= args.num_channel,
            ).to(args.device)


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
    # Load target classifier 

    train(wandb, args, classifier, classifier_val)
    wandb.finish()


if __name__ == '__main__':
    main()
