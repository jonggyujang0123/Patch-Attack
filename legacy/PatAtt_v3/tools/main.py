"""Import default libraries"""
import os
from models.resnet_32x32 import resnet10
from models.CGAN import Generator, Discriminator, Cutout
import torch.nn.functional as F
from utils.base_utils import set_random_seeds, get_accuracy, AverageMeter, WarmupCosineSchedule, WarmupLinearSchedule, load_ckpt, save_ckpt, get_data_loader, noisy_labels, accuracy
import torch.nn as nn
import torch
import wandb
import os
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from einops import rearrange
import argparse

def para_config():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    # hyperparameter setting
    parser.add_argument("--epochs",
            type=int,
            default=30)
    parser.add_argument("--test",
            type=bool,
            default=False)
    parser.add_argument("--train-batch-size",
            type=int,
            default= 64)
    parser.add_argument("--test-batch-size",
            type=int,
            default=100)
    parser.add_argument("--random-seed",
            type=int,
            default=0)
    parser.add_argument("--eval-every",
            type=int,
            default=10)
    parser.add_argument("--pin-memory",
            type=bool,
            default=True)

    # hyperparameter for generator and discrminator
    parser.add_argument("--patch-size",
            type=int,
            default=8)
    parser.add_argument("--patch-stride",
            type=int,
            default=2)
    parser.add_argument("--patch-padding",
            type=int,
            default=2)
    parser.add_argument("--n-gf",
            type=int,
            default=64,
            help="number of generating feature base")
    parser.add_argument("--n-df",
            type=int,
            default=192,
            help="number of discriminator feature base")
    parser.add_argument("--level-g",
            type=int,
            default=2,
            help="level of Conv layer in Generator")
    parser.add_argument("--level-d",
            type=int,
            default=4,
            help="level of Conv layer in Discrim")
    parser.add_argument("--latent-size",
            type=int,
            default=100)
    parser.add_argument("--w-attack",
            type=float,
            default=1.0)
    parser.add_argument("--w-mr",
            type=float,
            default=0.0)
    parser.add_argument("--gan-labelsmooth",
            type=float,
            default=0.0)
    parser.add_argument("--p-flip",
            type=float,
            default=0.0)
    parser.add_argument("--epoch-pretrain",
            type=int,
            default=-1)
    parser.add_argument("--x-sample",
            type=int,
            default=1)
    parser.add_argument('--transform', default=False, action=argparse.BooleanOptionalAction)

    # dataset 
    parser.add_argument("--aux-dataset",
            type=str,
            default="emnist",
            help = "choose one of mnist, emnist, fashion")
    parser.add_argument("--target-dataset",
            type=str,
            default="mnist",
            help = "choose one of mnist, emnist, fashion")
    parser.add_argument("--num-workers",
            type=int,
            default=4)
    # save path configuration
    parser.add_argument('--ckpt-dir',
            type=str,
            default='../experiments')
    # WANDB SETTINGS
    parser.add_argument("--wandb-project",
            type=str,
            default="PMIA-MNIST")
    parser.add_argument("--wandb-id",
            type=str,
            default="jonggyujang0123")
    parser.add_argument("--wandb-name",
            type=str,
            default="ResNet50")
    parser.add_argument("--wandb-active",
            type=bool,
            default=True)
    # optimizer setting 
    parser.add_argument("--weight-decay",
            type=float,
            default = 5.0e-4)
    parser.add_argument("--beta-1",
            type=float,
            default = 0.0)
    parser.add_argument("--beta-2",
            type=float,
            default = 0.9)
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
    parser.add_argument("--max-classes",
            type=int,
            default=10)

    args = parser.parse_args()
    
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.ckpt_path = os.path.join(args.ckpt_dir, 'classifier', args.target_dataset)
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

def train(wandb, args, classifier, classifier_val, G, D):
    cut_out = Cutout()
    fixed_z = torch.randn(
            args.test_batch_size,
            args.latent_size,
            ).to(args.device)

    fixed_c = torch.arange(
            min(args.num_classes, args.max_classes)
            ).unsqueeze(1).tile([1,args.test_batch_size//min(args.num_classes, args.max_classes)]).view(-1).to(args.device)
    BCE_loss = nn.BCELoss()
    CE_loss = nn.CrossEntropyLoss()
    optimizer_g = optim.Adam(G.parameters(), lr =args.lr, betas = (args.beta_1, args.beta_2))
    optimizer_d = optim.Adam(D.parameters(), lr =args.lr, betas = (args.beta_1, args.beta_2))
    train_loader, _, _ = get_data_loader(args= args, dataset_name= args.aux_dataset)
    if args.decay_type == "cosine":
        scheduler_g = WarmupCosineSchedule(optimizer_g, warmup_steps=args.warmup_steps, t_total=args.epochs*len(train_loader))
        scheduler_d = WarmupCosineSchedule(optimizer_d, warmup_steps=args.warmup_steps, t_total=args.epochs*len(train_loader))
    else:
        scheduler_g = WarmupLinearSchedule(optimizer_g, warmup_steps=args.warmup_steps, t_total=args.epochs*len(train_loader))
        scheduler_d = WarmupLinearSchedule(optimizer_d, warmup_steps=args.warmup_steps, t_total=args.epochs*len(train_loader))
    start_epoch = 0
    
    Loss_g = AverageMeter()
    Loss_d = AverageMeter()
    Loss_info = AverageMeter()
    D_x_total = AverageMeter()
    D_G_z_total = AverageMeter()
    Acc_total_t = AverageMeter()
    Max_act = AverageMeter()

    pbar = tqdm(
            range(start_epoch,args.epochs),
            bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
            )
#    pad = Pad(args.patch_size - args.patch_stride, fill=0)
    # Prepare dataset and dataloader
    for epoch in pbar:
        # Switch Training Mode
        G.train()
        D.train()
#        w_attack = np.clip( epoch / args.epochs *10, 0.0, 1.0) * args.w_attack
        w_attack = args.w_attack
        tepoch = tqdm(train_loader,
                bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
                )
        for data in tepoch:
            # (1) Update Discriminator (real data)
            optimizer_d.zero_grad()
            inputs = data[0].to(args.device, 
                    non_blocking=True)
            d_logit_t = D(inputs, transform = args.transform)
            real_target = torch.ones(d_logit_t.size(0)).to(args.device)  
#            real_target = noisy_labels(real_target, args.p_flip)
            fake_target = torch.zeros(d_logit_t.size(0)).to(args.device)
            fake_target = noisy_labels(fake_target, args.p_flip)
            err_real_d = BCE_loss(d_logit_t.view(-1),real_target - args.gan_labelsmooth)
            D_x = d_logit_t.view(-1).detach().mean().item()
            D_x_total.update(D_x, d_logit_t.size(0))
            
            # (2) update discrminator (fake data)

            latent_vector = torch.randn(inputs.size(0), args.latent_size).to(args.device)
            c = torch.multinomial(torch.ones(
                min(args.num_classes, args.max_classes)
                ), inputs.size(0), replacement=True).to(args.device)
            fake = G(latent_vector, c)

            d_logit_f = D(fake.detach(), transform = args.transform)
            err_fake_d = BCE_loss(d_logit_f.view(-1), fake_target + args.gan_labelsmooth)
            D_loss = err_real_d + err_fake_d

            D_loss.backward()

            optimizer_d.step()
            err_d = (err_fake_d + err_real_d)
            Loss_d.update(err_d.detach().mean().item(), d_logit_t.size(0))
            # (3) Generator update
            optimizer_g.zero_grad()
            outputs= D.forward(fake, transform = args.transform)
            err_g = BCE_loss(outputs.view(-1) , real_target)

            # (4) MI_loss Loss 
            D_disc = classifier(fake)
            #  D_disc = classifier(cut_out(fake))
            loss_attack = CE_loss(D_disc, c)
            info_loss = (w_attack * loss_attack) #/ D.n_patches
            mr_loss = - args.w_mr * ( D_disc.max(1)[0].mean() )
            if epoch > args.epoch_pretrain:
                total_loss = err_g + info_loss + mr_loss #+ mr_loss_avg
                #  total_loss = info_loss + mr_loss #+ mr_loss_avg
            else:
                total_loss = err_g
            total_loss.backward()
            optimizer_g.step()

            scheduler_d.step()
            scheduler_g.step()

            train_acc_target = D_disc.softmax(1)[torch.arange(c.size(0)),c].mean().item() 

            Acc_total_t.update(train_acc_target, inputs.size(0))
            Loss_g.update(err_g.detach().item(), inputs.size(0))
            Loss_info.update(info_loss.detach().item(), inputs.size(0))
            D_G_z_total.update(outputs.mean().item(), inputs.size(0))
            Max_act.update(D_disc.max(1)[0].mean().item(), inputs.size(0))
            tepoch.set_description(f'Ep {epoch}: L_D : {Loss_d.avg:2.3f}, L_G: {Loss_g.avg:2.3f}, L_info: {Loss_info.avg:2.3f}, lr: {scheduler_d.get_lr()[0]:.1E}, Acc:{Acc_total_t.avg:2.3f}, Max_A: {Max_act.avg:2.1f}')
        # (5) After end of epoch, save result model
        if epoch % args.eval_every == args.eval_every - 1 or epoch==0:
            _, images = evaluate(wandb, args, classifier, G, fixed_z=fixed_z, fixed_c = fixed_c, epoch = epoch)
            model_to_save_g = G.module if hasattr(G, 'module') else G
            model_to_save_d = D.module if hasattr(D, 'module') else D
            ckpt = {
                    'model_g' : model_to_save_g.state_dict(),
                    'model_d' : model_to_save_d.state_dict(),
                    }
            save_ckpt(checkpoint_fpath = args.ckpt_dir + f'/PMI/{args.target_dataset}_{args.patch_size}', checkpoint = ckpt, is_best=True)
            if args.wandb_active:
                wandb.log({
                    "loss_D" : Loss_d.avg,
                    "Loss_G" : Loss_g.avg,
                    "Loss_Info" : Loss_info.avg,
                    "D(x)" : D_x_total.avg,
                    "D(G(z))" : D_G_z_total.avg,
                    "val_acc_t" : Acc_total_t.avg,
                    "image" : images,
                    },
                    step = epoch)
        Loss_g.reset()
        Loss_d.reset()
        Loss_info.reset()
        D_G_z_total.reset()
        D_x_total.reset()
        Acc_total_t.reset()
    #  test(wandb, args, classifier_val, G)


def evaluate(wandb, args, classifier, G, fixed_z=None, fixed_c = None, epoch = 0):
    G.eval()
    if fixed_z == None:
        fixed_z = torch.randn(
                args.test_batch_size, 
                args.latent_size,
                ).to(args.device)
    if fixed_c == None:
        fixed_c = torch.multinomial(
                torch.ones(min(args.num_classes, args.max_classes)), 
                args.test_batch_size, replacement=True).to(args.device)
    val_acc = 0 
    fake = G(fixed_z, fixed_c)
    pred = classifier( fake)
    fake_y = fixed_c
    val_acc = (pred.max(1)[1] == fake_y).float().mean().item()
    arg_sort = pred[:, fixed_c[0]].argsort(descending=True)
    fake = fake[arg_sort, ...]
    fake = fake.reshape(
            10, 10,
            #  min(args.num_classes, args.max_classes),
            #  args.test_batch_size//min(args.num_classes, args.max_classes),
            args.num_channel, args.img_size, args.img_size)
    if args.num_channel == 1:
        fake = F.pad(fake, (1,1,1,1), value=1)
    else:
        fake = F.pad(fake, (1,1,1,1), value=-1)
    
    image_array = rearrange(fake, 
            'b1 b2 c h w -> (b2 h) (b1 w) c').cpu().detach().numpy().astype(np.float64)
    images = wandb.Image(image_array, caption=f'Epoch: {epoch}, Acc: {val_acc*100:2.2f}')
    return val_acc, images

def test(wandb, args, classifier_val, G):
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics import StructuralSimilarityIndexMeasure
    #  from sentence_transformers import SentenceTransformer, util
    #  model = SentenceTransformer('clip-ViT-B-32', device = args.device)
    #  import torchvision.transforms as T
    #  from PIL import Image
    #  transform = T.ToPILImage()
    G.eval()
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
            fixed_z = torch.randn(
                    y.shape[0],
                    args.latent_size,
                    ).to(args.device)
            label = y.to(args.device)
            fake = G(fixed_z, label)
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
                   name = f'{args.target_dataset}_P{args.patch_size}, S{args.patch_stride}, w_att:{args.w_attack}, n_df:{args.n_df}, lr:{args.lr}',
                   group = f'{args.target_dataset}_Patch: {args.patch_size}'
                   )
    else:
        os.environ['WANDB_MODE'] = 'dryrun'
    print(args)
    # set cuda flag
    if not torch.cuda.is_available():
        print("WARNING: You have a CUDA device, so you should probably enable CUDA")
    # We need to use seeds to make sure that model initialization same
    set_random_seeds(random_seed = args.random_seed)
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    
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
    G = Generator(
            img_size = args.img_size,
            latent_size = args.latent_size,
            levels= args.level_g,
            num_classes = min(args.num_classes, args.max_classes),
            n_gf = args.n_gf,
            n_c = args.num_channel,
            ).to(args.device)
    D = Discriminator(
            img_size = args.img_size,
            patch_size = args.patch_size,
            patch_stride= args.patch_stride,
            patch_padding = args.patch_padding,
            n_df = args.n_df,
            n_c = args.num_channel, 
            ).to(args.device)


    if os.path.exists(args.ckpt_path):
        ckpt = load_ckpt(args.ckpt_path, is_best = False)
        classifier.load_state_dict(ckpt['model'])
        classifier.eval()
        print(f'{args.ckpt_path} model is loaded!')
    else:
        raise Exception('there is no generative checkpoint')
    
    fpath_val = args.ckpt_path + '_valid'
    if os.path.exists(fpath_val):
        ckpt = load_ckpt(fpath_val, is_best = False)
        classifier_val.load_state_dict(ckpt['model'])
        classifier_val.eval()
        print(f'{fpath_val} model is loaded!')
    else:
        raise Exception('there is no generative checkpoint')
    # Load target classifier 
    if args.test:
        ckpt = load_ckpt(args.ckpt_dir + f'/PMI/{args.target_dataset}_{args.patch_size}', is_best = args.test)
        G.load_state_dict(ckpt['model_g'])
        D.load_state_dict(ckpt['model_d'])

    if args.test:
        test(wandb, args,  classifier_val, G)
    else:
        train(wandb, args, classifier, classifier_val, G, D)
    wandb.finish()


if __name__ == '__main__':
    main()
