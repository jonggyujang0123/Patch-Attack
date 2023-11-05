"""Import default libraries"""
import os
import torch.nn.functional as F
from models.CGAN import Generator, Discriminator, Qrator
from utils.base_utils import set_random_seeds, get_accuracy, AverageMeter, WarmupCosineSchedule, WarmupLinearSchedule, load_ckpt, save_ckpt, get_data_loader, noisy_labels, accuracy, CutMix, Mixup, sample_noise, Sharpness, Cutout, AddGaussianNoise
import torch.nn as nn
import torch
import wandb
import os
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from einops import rearrange
import argparse
import torchvision.transforms as transforms
import itertools
from models.resnet_32x32 import ResNet10, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from models.densenet import DenseNet121, DenseNet161, DenseNet169, DenseNet201
from models.dla import DLA
def para_config():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    # hyperparameter setting
    parser.add_argument("--off-aug",
            action=argparse.BooleanOptionalAction,
            default=False)
    parser.add_argument("--off-patch",
            action=argparse.BooleanOptionalAction,
            default=False)
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
    parser.add_argument("--visualize-nrow",
            type=int,
            default=10)
    parser.add_argument("--random-seed",
            type=int,
            default=0)
    parser.add_argument("--eval-every",
            type=int,
            default=1)
    parser.add_argument("--pin-memory",
            type=bool,
            default=True)
    # hyperparameter for generator and discrminator
    parser.add_argument("--n-gf",
            type=int,
            default=64,
            help="number of generating feature base")
    parser.add_argument("--n-df",
            type=int,
            default=256,
            help="number of discriminator feature base")
    parser.add_argument("--patch-size",
            type=int,
            default=4,
            help="patch size of discriminator")
    parser.add_argument("--patch-stride",
            type=int,
            default=2,
            help="patch stride of discriminator")
    parser.add_argument("--patch-padding",
            type=int,
            default=0,
            help="patch padding of discriminator")
    parser.add_argument("--n-qf",
            type=int,
            default=64,
            help="number of info feature base")
    parser.add_argument("--level-q",
            type=int,
            default=3,
            help="level of Conv layer in Info")
    parser.add_argument("--level-g",
            type=int,
            default=3,
            help="level of Conv layer in Generator")
    parser.add_argument("--level-d",
            type=int,
            default=3,
            help="level of Conv layer in Discrim")
    parser.add_argument("--latent-size",
            type=int,
            default=100)
    parser.add_argument("--target-class",
            type=int,
            default=1)
    parser.add_argument("--n-disc",
            type=int,
            default=1,
            help="number of discriminator")
    parser.add_argument("--w-attack",
            type=float,
            default=1.0)
    parser.add_argument("--w-disc",
            type=float,
            default=1.0)
    parser.add_argument("--gan-labelsmooth",
            type=float,
            default=0.0)
    parser.add_argument("--epoch-pretrain",
            type=int,
            default=-1)
    parser.add_argument("--n-images",
            type=int,
            default=1000)

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
    parser.add_argument("--beta-1",
            type=float,
            default = 0.5)
    parser.add_argument("--beta-2",
            type=float,
            default = 0.999)
    parser.add_argument("--lr",
            type=float,
            default=2e-4,
            help = "learning rate")
    args = parser.parse_args()
    
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.ckpt_path = os.path.join(args.ckpt_dir, 'classifier', args.target_dataset)
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

    if args.off_patch:
        args.patch_size = args.img_size
        args.patch_stride = args.img_size
        args.patch_padding = 0
        args.n_df = 1024
    return args

class AUGMENT_FWD(nn.Module):
    def __init__(
            self,
            args,
            ):
        super(AUGMENT_FWD, self).__init__()
        if args.target_dataset in ['mnist']:
            self.trans = torch.nn.Sequential(
                    transforms.RandomRotation(20, fill=-1, expand=True),
                    transforms.RandomResizedCrop(args.img_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
                    #  transforms.RandomAffine(
                    #      0,
                    #      translate=(0.1, 0.1),
                    #      fill=-1,
                        #  )
                    )
            self.iterations=2
            self.mean = False
        elif args.target_dataset in ['emnist']:
            self.trans = torch.nn.Sequential(
                    transforms.RandomRotation(20, fill=-1, expand=True),
                    transforms.RandomResizedCrop(args.img_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
                    #  transforms.RandomAffine(
                    #      0,
                    #      translate=(0.1, 0.1),
                    #      fill=-1,
                    #      )
                    #  transforms.RandomCrop(args.img_size),
                    )
            self.iterations=2
            self.mean =False

        elif args.target_dataset in ['cifar10']:
            self.trans = torch.nn.Sequential(
                    transforms.Pad(4, fill=0),
                    transforms.RandomRotation(22.5, fill=0, expand=False),
                    transforms.RandomRotation(22.5, fill=0, expand=False),
                    transforms.RandomHorizontalFlip(p=0.5),
                    #  transforms.RandomCrop(args.img_size),
                    #  transforms.RandomResizedCrop(args.img_size, scale=(0.7, 0.8), ratio=(1.0, 1.0)),
                    #  transforms.RandomAffine(
                    #      0,
                    #      translate=(0.1, 0.1),
                    #      )
                    transforms.RandomCrop(args.img_size),
                    )
            self.iterations=4
            self.mean=True
        else:
            raise NotImplementedError
        print(self.trans)

    def forward(self, classifier, x):
        b, c, h, w = x.shape
        #  x = [self.trans(x) for _ in range(self.iterations)]
        #  if torch.rand(1) < 0.5:
            #  x = self.trans(x)
        #  x = [self.trans(x) if torch.rand(1) < 0.5 else x for _ in range(self.iterations)]
        #  for _ in range(self.iterations):
            #  x_list.append(x if
        x = [x]+ [self.trans(x) for _ in range(self.iterations-1)]

        x = torch.cat(x)
        x = classifier(x)
        if self.mean:
            x = x.view(self.iterations, b, -1).mean(dim=0)
        #  x = x.softmax(dim=-1).log()
        #  x = x.view(self.iterations, b, -1).mean(dim=0)

        #  x =  x.view(self.iterations, b, -1).softmax(dim=-1).log().min(dim=0)[0]
        #  x =  (x.view(self.iterations, b, -1).softmax(dim=-1)+1e-6).log().mean(dim=0)
        return x

def train(wandb, args, classifier, classifier_val, G, D, Q):
    from torch.cuda.amp import autocast
    from torch.cuda.amp import GradScaler
    scaler = GradScaler()
    aug = AUGMENT_FWD(
            args = args,
            ).to(args.device)
    fixed_z, _ = sample_noise(
            n_disc = args.n_disc,
            n_z = args.latent_size,
            batch_size = args.test_batch_size,
            device = args.device,
            )
    #  GAN_loss = nn.MSELoss()
    GAN_loss = nn.BCEWithLogitsLoss()
    #  Qrator_loss = nn.NLLLoss()
    #  Attacker_loss= nn.NLLLoss()
    Qrator_loss = nn.CrossEntropyLoss() 
    Attacker_loss= nn.CrossEntropyLoss()
    #  optimizer_g = optim.Adam(G.parameters(), lr =args.lr, betas = (args.beta_1, args.beta_2))
    optimizer_d = optim.Adam(D.parameters(), lr =args.lr, betas = (args.beta_1, args.beta_2))
    optimizer_info = optim.Adam(
            itertools.chain(Q.parameters(), G.parameters()),
            lr =args.lr, betas = (args.beta_1, args.beta_2))
            
    train_loader, _, _ = get_data_loader(args= args, dataset_name= args.aux_dataset)
    #  scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=args.epochs)
    #  scheduler_info = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_info, T_max=args.epochs)
    start_epoch = 0
    
    Loss_g = AverageMeter()
    Loss_d = AverageMeter()
    Loss_q = AverageMeter()
    D_x_total = AverageMeter()
    D_G_z_total = AverageMeter()
    Acc_total_t = AverageMeter()
    Max_act = AverageMeter()

    pbar = tqdm(
            range(start_epoch,args.epochs),
            bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}',
            disable = True,
            )
    # Prepare dataset and dataloader
    for epoch in pbar:
        # Switch Training Mode
        G.train()
        D.train()
        Q.train()
        tepoch = tqdm(train_loader,
                bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}',
                #  leave = False,
                )
        for data in tepoch:
            # (1) Update Discriminator (real data)
            optimizer_d.zero_grad()
            x_real= data[0].to(args.device, 
                    non_blocking=True)
            #  with autocast():
            d_logit_t = D(x_real)
            real_target = torch.ones(d_logit_t.size()).to(args.device).float()
            fake_target = torch.zeros(d_logit_t.size()).to(args.device).float()
            err_real_d = GAN_loss(d_logit_t,real_target - args.gan_labelsmooth)
            #  err_real_d = GAN_loss(d_logit_t,real_target)
            D_x = d_logit_t.view(-1).detach().mean().item()
            D_x_total.update(D_x, d_logit_t.size(0))
            
            # (2) update discrminator (fake data)

            latent_vector, idx = sample_noise(
                    n_disc = args.n_disc,
                    n_z = args.latent_size,
                    batch_size = x_real.size(0),
                    device = args.device,
                    )
            c = torch.LongTensor(x_real.size(0)).fill_(args.target_class).to(args.device)
            x_fake = G(latent_vector)
            d_logit_f = D(x_fake.detach())
            err_fake_d = GAN_loss(d_logit_f, fake_target)
            D_loss = err_real_d + err_fake_d

            D_loss.backward()
            optimizer_d.step()
            #  scaler.scale(D_loss).backward()
            #  scaler.step(optimizer_d)
            #  scaler.update()

            err_d = D_loss.detach().mean().item()
            Loss_d.update(err_d, d_logit_t.size(0))
            
            # (3) Update Qrator and Generator
            optimizer_info.zero_grad()

            outputs= D(x_fake)
            err_g = GAN_loss(outputs , real_target)
        
            q_loss = 0.0
            if args.n_disc != 1:
                #  logit_q = Q(x_fake).view(x_fake.size(0), -1)
                if args.off_aug:
                    logit_q = Q(x_fake)
                else:
                    logit_q = aug(Q, x_fake)
                
                q_loss += Qrator_loss(logit_q, idx.view(-1).tile(logit_q.size(0)//idx.size(0))) * args.w_disc
                #  q_loss += CE_loss(logit_q.reshape(-1, args.n_disc), idx.view(-1)) * args.w_disc
            if args.off_aug:
                logit_att = classifier(x_fake)
            else:
                logit_att= aug(classifier, x_fake)
            loss_att = args.w_attack * Attacker_loss(logit_att, c.view(-1).tile(logit_att.size(0)//c.size(0)))
            #  loss_att = args.w_attack * NLL_loss(logit_att, c)

            
            if epoch < args.epoch_pretrain:
                total_loss = err_g + q_loss
            else:
                total_loss = err_g + loss_att + q_loss 

            total_loss.backward()
            optimizer_info.step()
            #  scaler.scale(total_loss).backward()
            #  scaler.step(optimizer_info)
            #  scaler.update()
            train_acc_target = logit_att.softmax(1).mean(dim=0)[args.target_class].item()
            #  train_acc_target = logit_att.softmax(1)[range(logit_att.size(0)), c].mean().item()

            Acc_total_t.update(train_acc_target, x_real.size(0))
            Loss_g.update(err_g.detach().item(), x_real.size(0))
            Loss_q.update(q_loss.detach().item() if q_loss != 0 else 0, x_real.size(0))
            D_G_z_total.update(outputs.mean().item(), x_real.size(0))
            Max_act.update(loss_att.mean().item() / ( args.w_attack +1e-5), x_real.size(0))
            #  tepoch.set_description(f'Ep {epoch}: L_D: {Loss_d.avg:2.3f}, L_G: {Loss_g.avg:2.3f}, L_Q: {Loss_q.avg:2.3f}, lr: {scheduler_d.get_lr()[0]:.1E}, Acc:{Acc_total_t.avg:2.3f}, MR: {Max_act.avg:2.1f}')
            tepoch.set_description(f'Ep {epoch}: L_D: {Loss_d.avg:2.3f}, L_G: {Loss_g.avg:2.3f}, L_Q: {Loss_q.avg:2.3f}, Acc:{Acc_total_t.avg:2.3f}, MR: {Max_act.avg:2.1f}')
        # (5) After end of epoch, save result model
        #  scheduler_d.step()
        #  scheduler_info.step()
        if epoch % args.eval_every == args.eval_every - 1 or epoch==0:
            _, images = evaluate(wandb, args, classifier_val, G, D, fixed_z=fixed_z, epoch = epoch)
            model_to_save_g = G.module if hasattr(G, 'module') else G
            model_to_save_d = D.module if hasattr(D, 'module') else D
            ckpt = {
                    'model_g' : model_to_save_g.state_dict(),
                    'model_d' : model_to_save_d.state_dict(),
                    }
            save_ckpt(checkpoint_fpath = args.ckpt_dir + f'/PMI/{args.target_dataset}', checkpoint = ckpt, is_best=True)
            if args.wandb_active:
                wandb.log({
                    "loss_D" : Loss_d.avg,
                    "Loss_G" : Loss_g.avg,
                    "Loss_Q" : Loss_q.avg,
                    "D(x)" : D_x_total.avg,
                    "D(G(z))" : D_G_z_total.avg,
                    "val_acc_t" : Acc_total_t.avg,
                    "image" : images,
                    },
                    step = epoch)
        Loss_g.reset()
        Loss_d.reset()
        Loss_q.reset()
        D_G_z_total.reset()
        D_x_total.reset()
        Acc_total_t.reset()
    return G
    #  test(wandb, args, classifier_val, G)

def save_images(args, G):
    from torchvision.utils import save_image
    import torchvision
    z, _ = sample_noise(
            n_disc = args.n_disc,
            n_z = args.latent_size,
            batch_size = args.n_images,
            device = args.device,
            )
    fake = (G(z) + 1)/2.0
    dir_name = 'Patch_MI'
    if args.off_patch:
        dir_name = dir_name + '_32'
    if args.off_aug:
        dir_name = dir_name + '_noaug'
    directory = f'./Results/{dir_name}/{args.target_dataset}/{args.target_class}'


    if not os.path.exists(directory):
        os.makedirs(directory)
    for i in range(args.n_images):
        tensor = fake[i, ...].cpu().detach()
        if args.num_channel == 1:
            tensor = torch.cat([tensor, tensor, tensor], dim = 0)
        save_image(tensor, f'{directory}/{i}.png')

def evaluate(wandb, args, classifier, G, D, fixed_z=None, epoch = 0):
    G.eval()
    if fixed_z is None:
        fixed_z, _ = sample_noise(
                n_disc=args.n_disc,
                n_z = args.latent_size,
                batch_size=args.test_batch_size,
                device=args.device,
                )
    val_acc = 0 
    x_fake = G(fixed_z)
    pred = classifier( x_fake)
    val_acc = (pred.max(1)[1] == args.target_class).float().mean().item()
    #  arg_sort = (torch.log(D(x_fake)[:,0] + 1e-8) + args.w_attack*torch.log(pred.softmax(dim=1)[:, fixed_c[0]])).argsort(descending=True)
    #  x_fake = x_fake[arg_sort, ...]
    x_fake = x_fake.reshape(
            args.test_batch_size//args.visualize_nrow, args.visualize_nrow,
            args.num_channel, args.img_size, args.img_size)
    if args.num_channel == 1:
        x_fake = F.pad(x_fake, (1,1,1,1), value=1)
    else:
        x_fake = F.pad(x_fake, (1,1,1,1), value=-1)
    
    image_array = rearrange(x_fake, 
            'b1 b2 c h w -> (b2 h) (b1 w) c').cpu().detach().numpy().astype(np.float64)
    images = wandb.Image(image_array, caption=f'Epoch: {epoch}, Acc: {val_acc*100:2.2f}')
    return val_acc, images


def main():
    args = para_config()
    if args.wandb_active:
        wandb.init(project = args.wandb_project,
                   entity = args.wandb_id,
                   config = args,
                   name = f'{args.target_dataset} w_att:{args.w_attack}, lr:{args.lr}, Patch_size: {args.patch_size}, stride: {args.patch_stride}, padding: {args.patch_padding}',
                   group = f'{args.target_dataset}'
                   )
    else:
        os.environ['WANDB_MODE'] = 'dryrun'
    print(args)
    set_random_seeds(random_seed = args.random_seed)
    

    classifier = ResNet18(num_classes = args.num_classes, num_channel = args.num_channel).to(args.device)
    classifier_val = DLA(num_classes = args.num_classes, num_channel = args.num_channel).to(args.device)
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
    

    G = Generator(
            img_size = args.img_size,
            latent_size = args.latent_size,
            levels= args.level_g,
            n_gf = args.n_gf,
            n_c = args.num_channel,
            n_disc= args.n_disc,
            ).to(args.device)
    print(G)
    D = Discriminator(
            img_size = args.img_size,
            patch_size = args.patch_size,
            patch_stride = args.patch_stride,
            patch_padding = args.patch_padding,
            #  scale_factor = args.scale_factor,
            #  kernel_size = args.d_ksize,
            levels= args.level_d,
            n_df = args.n_df,
            n_c = args.num_channel,
            ).to(args.device)
    print(D)
    Q = Qrator(
            img_size= args.img_size, 
            n_qf = args.n_qf,
            levels = args.level_q,
            n_c = args.num_channel,
            n_disc= args.n_disc,
            ).to(args.device)
    print(Q)

    if args.test:
        ckpt = load_ckpt(args.ckpt_dir + f'/PMI/{args.target_dataset}', is_best = args.test)
        G.load_state_dict(ckpt['model_g'])
        D.load_state_dict(ckpt['model_d'])

    G = train(wandb, args, classifier, classifier_val, G, D, Q)
    save_images(args, G)

    wandb.finish()


if __name__ == '__main__':
    main()
