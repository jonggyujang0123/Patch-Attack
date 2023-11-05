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
from torchvision.models import densenet, inception, resnet
from models.resnet import resnet10, resnet18, resnet34, resnet50, resnet101
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
            default=64,
            help="number of discriminator feature base")
    #  parser.add_argument("--scale-factor",
    #          type=int,
    #          default=4,
    #          help="scale factor of discriminator")
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
            default=2,
            help="level of Conv layer in Info")
    parser.add_argument("--level-g",
            type=int,
            default=2,
            help="level of Conv layer in Generator")
    parser.add_argument("--level-d",
            type=int,
            default=3,
            help="level of Conv layer in Discrim")
    #  parser.add_argument("--d-ksize"
    #          ,type=int,
    #          default=4,
    #          help="kernel size of discriminator")
    parser.add_argument("--latent-size",
            type=int,
            default=100)
    parser.add_argument("--target-class",
            type=int,
            default=1)
    parser.add_argument("--n-disc",
            type=int,
            default=2,
            help="number of discriminator")
    parser.add_argument("--n-cont",
            type=int,
            default=0,
            help="number of continuous variable")
    parser.add_argument("--w-attack",
            type=float,
            default=1.0)
    parser.add_argument("--w-mr",
            type=float,
            default=0.0)
    parser.add_argument("--w-disc",
            type=float,
            default=1.0)
    parser.add_argument("--w-cont",
            type=float,
            default=0.3)
    parser.add_argument("--gan-labelsmooth",
            type=float,
            default=0.0)
    parser.add_argument("--epoch-pretrain",
            type=int,
            default=-1)

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
    parser.add_argument("--decay-type",
            type=str,
            default="linear",
            help="choose linear or cosine")
    parser.add_argument("--warmup-steps",
            type=int,
            default=100)
    parser.add_argument("--lr",
            type=float,
            default=2e-4,
            help = "learning rate")
    args = parser.parse_args()
    
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.ckpt_path = os.path.join(args.ckpt_dir, 'classifier', args.target_dataset)
    if args.target_dataset in ['mnist', 'fashion', 'kmnist', 'HAN']:
        args.num_channel = 1
        args.img_size = 32
        args.num_classes = 10
    if args.target_dataset == 'emnist':
        args.num_channel = 1
        args.img_size = 32
        args.num_classes = 26
    if args.target_dataset == 'cifar10':
        args.num_channel = 3
        args.img_size = 64
        args.num_classes = 10
    if args.target_dataset == 'cifar100':
        args.num_channel = 3
        args.img_size = 32
        args.num_classes = 100
    if args.target_dataset == 'LFW':
        args.num_channel = 3
        args.img_size = 64
        args.num_classes = 10
    if args.dataset in 'celeba':
        args.num_channel = 3
        args.img_size = 128
        args.num_classes = 300


    return args

class AUGMENT_FWD(nn.Module):
    def __init__(
            self,
            args,
            ):
        super(AUGMENT_FWD, self).__init__()
        if args.target_dataset in ['mnist', 'emnist', 'fashion']:
            self.trans = torch.nn.Sequential(
                    #  CutMix(),
                    #  Mixup(),
                    transforms.RandomRotation(15, fill=-1, expand=True),
                    transforms.RandomRotation(15, fill=-1, expand=False),
                    transforms.RandomResizedCrop(args.img_size, scale=(0.75, 1.0), ratio=(1.0, 1.0)),
                    transforms.RandomAffine(
                        degrees=0,
                        translate=(0.1, 0.1),
                        fill=-1),
                    transforms.Resize(args.img_size),
                    )
            self.iterations=8
        elif args.target_dataset in ['celeba', 'cinic10', 'cifar10', 'cifar100', 'LFW']:
            self.trans = torch.nn.Sequential(
                    transforms.RandomAffine(
                        degrees=10,
                        scale=(0.8, 1.2),
                        fill=0),
                    #  Cutout(
                    #      n_holes=3,
                    #      length=0.25,
                    #      fill=0,
                    #      ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomResizedCrop(args.img_size, scale=(0.75, 1.0), ratio=(1.0, 1.0)),
                    transforms.Resize(args.img_size),
                    )
            self.iterations=4
        print(self.trans)

    def forward(self, x):
        #  b, c, h, w = x.size()
        #  x = [self.trans(x) for _ in range(self.iterations)]
        x = [x] + [self.trans(x) for _ in range(self.iterations-1)]
        #  x = torch.cat([x, x.view(self.iterations-1, b, c, h, w).mean(dim=0)])
        x = torch.cat(x)
        ## Results 
        #  x = self.classifier(torch.cat(x)).view(self.iterations, b, -1).mean(dim=0)
        #  x = model(torch.cat(x)).view(self.iterations, b, -1)
        #  x_mr = x.mean(dim=0)
        #  x = x.softmax(dim=-1).mean(dim=0).log()
        #  x = x.mean(dim=0).softmax(dim=-1).log()
        #  x = torch.log(x/(1-x))
        return x

def train(wandb, args, classifier, classifier_val, G, D, Q):
    aug = AUGMENT_FWD(
            args = args,
            ).to(args.device)
    #  cutmix = CutMix(
            #  ).to(args.device)
    fixed_z, _ = sample_noise(
            n_disc = args.n_disc,
            n_cont = args.n_cont,
            n_z = args.latent_size,
            batch_size = args.test_batch_size,
            device = args.device,
            )
    GAN_loss = nn.MSELoss()
    CE_loss = nn.CrossEntropyLoss()
    NLL_loss = nn.NLLLoss()
    MSE_loss = nn.MSELoss()
    #  optimizer_g = optim.Adam(G.parameters(), lr =args.lr, betas = (args.beta_1, args.beta_2))
    optimizer_d = optim.Adam(D.parameters(), lr =args.lr, betas = (args.beta_1, args.beta_2))
    optimizer_info = optim.Adam(
            itertools.chain(Q.parameters(), G.parameters()),
            lr =args.lr, betas = (args.beta_1, args.beta_2))
            
    train_loader, _, _ = get_data_loader(args= args, dataset_name= args.aux_dataset)
    if args.decay_type == "cosine":
        #  scheduler_g = WarmupCosineSchedule(optimizer_g, warmup_steps=args.warmup_steps, t_total=args.epochs*len(train_loader))
        scheduler_d = WarmupCosineSchedule(optimizer_d, warmup_steps=args.warmup_steps, t_total=args.epochs*len(train_loader))
        scheduler_info = WarmupCosineSchedule(optimizer_info, warmup_steps=args.warmup_steps, t_total=args.epochs*len(train_loader))
    else:
        #  scheduler_g = WarmupLinearSchedule(optimizer_g, warmup_steps=args.warmup_steps, t_total=args.epochs*len(train_loader))
        scheduler_d = WarmupLinearSchedule(optimizer_d, warmup_steps=args.warmup_steps, t_total=args.epochs*len(train_loader))
        scheduler_info = WarmupLinearSchedule(optimizer_info, warmup_steps=args.warmup_steps, t_total=args.epochs*len(train_loader))
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
        #  w_attack = np.clip( epoch / args.epochs, 0.0, 1.0) * args.w_attack
        w_attack = args.w_attack
        tepoch = tqdm(train_loader,
                bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}',
                #  leave = False,
                )
        for data in tepoch:
            # (1) Update Discriminator (real data)
            optimizer_d.zero_grad()
            x_real= data[0].to(args.device, 
                    non_blocking=True)
            d_logit_t = D(x_real)
            real_target = torch.ones(d_logit_t.size()).to(args.device)
            fake_target = torch.zeros(d_logit_t.size()).to(args.device)
            err_real_d = GAN_loss(d_logit_t,real_target - args.gan_labelsmooth)
            #  err_real_d = GAN_loss(d_logit_t,real_target)
            D_x = d_logit_t.view(-1).detach().mean().item()
            D_x_total.update(D_x, d_logit_t.size(0))
            
            # (2) update discrminator (fake data)

            latent_vector, idx = sample_noise(
                    n_disc = args.n_disc,
                    n_cont = args.n_cont,
                    n_z = args.latent_size,
                    batch_size = x_real.size(0),
                    device = args.device,
                    )
            c = torch.LongTensor(x_real.size(0)).fill_(args.target_class).to(args.device)
            x_fake = G(latent_vector)
            d_logit_f = D(x_fake.detach(), real=False)
            #  err_fake_d = GAN_loss(d_logit_f, fake_target)
            err_fake_d = GAN_loss(d_logit_f, fake_target + args.gan_labelsmooth)
            D_loss = err_real_d + err_fake_d

            D_loss.backward()
            optimizer_d.step()
            scheduler_d.step()
            err_d = (err_fake_d + err_real_d)
            Loss_d.update(err_d.detach().mean().item(), d_logit_t.size(0))
            
            # (3) Update Qrator and Generator
            optimizer_info.zero_grad()
            x_fake_aug = aug(x_fake)
            q = Q(x_fake).view(x_fake.size(0), -1)
            #  logit_q = q.softmax(dim=-1).mean(dim=0).log()
            
            logit_q = q[:, :args.n_disc]
            q_cont = q[:, args.n_disc:]

            outputs= D(x_fake, real=False)
            err_g = GAN_loss(outputs , real_target)
            
            q_loss = 0.0
            if args.n_disc != 0:
                q_loss += CE_loss(logit_q.reshape(-1, args.n_disc), idx.view(-1)) * args.w_disc
            if args.n_cont != 0:
                q_loss += MSE_loss(q_cont, latent_vector[:, args.latent_size + args.n_disc:]) * args.w_cont
            cls = classifier(x_fake_aug).view(aug.iterations, x_fake.size(0), -1)
            logit_att = cls.mean(dim=0).softmax(dim=-1)[:,args.target_class].log()
            mr_att = cls.mean(dim=0)
#
            #  loss_att = w_attack * CE_loss(mr_att, c)
            loss_att = w_attack * (-logit_att).mean()   #w_attack * NLL_loss(logit_att, c)
            
            if epoch < args.epoch_pretrain:
                total_loss = err_g + q_loss
            else:
                total_loss = err_g + loss_att + q_loss 
            total_loss.backward()
            optimizer_info.step()
            scheduler_info.step()
            train_acc_target = mr_att.softmax(1)[range(mr_att.size(0)), c].mean().item()
            #  train_acc_target = logit_att.softmax(1)[range(logit_att.size(0)), c].mean().item()

            Acc_total_t.update(train_acc_target, x_real.size(0))
            Loss_g.update(err_g.detach().item(), x_real.size(0))
            Loss_q.update(q_loss.detach().item(), x_real.size(0))
            D_G_z_total.update(outputs.mean().item(), x_real.size(0))
            Max_act.update(loss_att.mean().item() / w_attack, x_real.size(0))
            tepoch.set_description(f'Ep {epoch}: L_D: {Loss_d.avg:2.3f}, L_G: {Loss_g.avg:2.3f}, L_Q: {Loss_q.avg:2.3f}, lr: {scheduler_d.get_lr()[0]:.1E}, Acc:{Acc_total_t.avg:2.3f}, MR: {Max_act.avg:2.1f}')
        # (5) After end of epoch, save result model
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
    #  test(wandb, args, classifier_val, G)


def evaluate(wandb, args, classifier, G, D, fixed_z=None, epoch = 0):
    G.eval()
    if fixed_z == None:
        fixed_z, _ = sample_noise(
                n_disc=args.n_disc,
                n_cont=args.n_cont,
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
    # set cuda flag
    if not torch.cuda.is_available():
        print("WARNING: You have a CUDA device, so you should probably enable CUDA")
    # We need to use seeds to make sure that model initialization same
    set_random_seeds(random_seed = args.random_seed)
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    
    if args.img_size == 32:
        from models.resnet import resnet18 as model
        from models.resnet import resnet50 as model_val
        classifier = model(
                num_classes= args.num_classes,
                num_channel= args.num_channel,
                ).to(args.device)
        classifier_val = model_val(
                num_classes= args.num_classes,
                num_channel= args.num_channel,
                ).to(args.device)
    else:
        classifier = resnet.resnet34(
                pretrained=True,
                )
        if args.num_channel == 1:
            classifier.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) 
        classifier.fc = nn.Linear(classifier.fc.in_features, args.num_classes)

        torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
        classifier_val = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
        classifier_val.fc = nn.Linear(classifier_val.fc.in_features, args.num_classes)

        classifier = classifier.to(args.device)
        classifier_val = classifier_val.to(args.device)
        #  from models.resnet import resnet34 as resnet
        #  from models.resnet import resnet50 as resnet_val

    G = Generator(
            img_size = args.img_size,
            latent_size = args.latent_size,
            levels= args.level_g,
            n_gf = args.n_gf,
            n_c = args.num_channel,
            n_disc= args.n_disc,
            n_cont = args.n_cont,
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
            n_cont = args.n_cont,
            ).to(args.device)
    print(Q)

    if os.path.exists(args.ckpt_path):
        ckpt = load_ckpt(args.ckpt_path, is_best = True)
        classifier.load_state_dict(ckpt['model'])
        classifier.eval()
        print(f'{args.ckpt_path} model is loaded!')
    else:
        raise Exception('there is no generative checkpoint')
    
    fpath_val = args.ckpt_path + '_valid'
    if os.path.exists(fpath_val):
        ckpt = load_ckpt(fpath_val, is_best = True)
        classifier_val.load_state_dict(ckpt['model'])
        classifier_val.eval()
        print(f'{fpath_val} model is loaded!')
    else:
        raise Exception('there is no generative checkpoint')
    # Load target classifier 
    if args.test:
        ckpt = load_ckpt(args.ckpt_dir + f'/PMI/{args.target_dataset}', is_best = args.test)
        G.load_state_dict(ckpt['model_g'])
        D.load_state_dict(ckpt['model_d'])

    if args.test:
        test(wandb, args,  classifier_val, G)
    else:
        train(wandb, args, classifier, classifier_val, G, D, Q)
    wandb.finish()


if __name__ == '__main__':
    main()
