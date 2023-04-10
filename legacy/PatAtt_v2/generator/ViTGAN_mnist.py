"""========Default import===="""
from __future__ import print_function
import argparse
from utils.base_utils import set_random_seeds, get_accuracy, AverageMeter, WarmupCosineSchedule, WarmupLinearSchedule, set_grad
import torch.nn as nn
import torch
import os
import torch.optim as optim
import shutil
from tqdm import tqdm
from easydict import EasyDict as edict
import yaml
import wandb
import torchvision.utils as vutils
import numpy as np
""" ==========END================"""

""" =========Configurable ======="""
from models.ViTGAN import Generator, Discriminator
#os.environ['WANDB_SILENT']='true'
""" ===========END=========== """



def parse_args():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config",
            type=str, 
            help="Configuration file in configs.")
    parser.add_argument("--resume",
            type=int,
            default= 0,
            help="Resume the last training from saved checkpoint.")
    parser.add_argument("--multigpu",
            type=bool, 
            help="Local rank. Necessary for using the torch.distributed.launch utility.",
            default= False)
    parser.add_argument("--test", 
            type=int,
            default = 0,
            help="if test, choose True.")
    args = parser.parse_args()
    return args

def get_data_loader(cfg, args):
    if cfg.dataset in ['cifar100', 'cifar10']:
        from datasets.cifar import get_loader_cifar as get_loader
    if cfg.dataset in ['mnist']:
        from datasets.mnist import get_loader_mnist as get_loader
    if cfg.dataset in ['emnist']:
        from datasets.emnist import get_loader_emnist as get_loader
    return get_loader(cfg, args)





def load_ckpt(checkpoint_fpath, is_best =False):
    """
    Latest checkpoint loader
        checkpoint_fpath : 
    :return: dict
        checkpoint{
            model,
            optimizer,
            epoch,
            scheduler}
    example :
    """
    if is_best:
        ckpt_path = checkpoint_fpath+'/'+'best.pt'
    else:
        ckpt_path = checkpoint_fpath+'/'+'checkpoint.pt'
    try:
        print(f"Loading checkpoint '{ckpt_path}'")
        checkpoint = torch.load(ckpt_path)
    except:
        print(f"No checkpoint exists from '{ckpt_path}'. Skipping...")
        print("**First time to train**")
    return checkpoint


def save_ckpt(checkpoint_fpath, checkpoint, is_best=False):
    """
    Checkpoint saver
    :checkpoint_fpath : directory of the saved file
    :checkpoint : checkpoiint directory
    :return:
    """
    ckpt_path = checkpoint_fpath+'/'+'checkpoint.pt'
    # Save the state
    if not os.path.exists(checkpoint_fpath):
        os.makedirs(checkpoint_fpath)
    torch.save(checkpoint, ckpt_path)
    # If it is the best copy it to another file 'model_best.pth.tar'
#    print("Checkpoint saved successfully to '{}' at (epoch {})\n"
#        .format(ckpt_path, checkpoint['epoch']))
    if is_best:
        ckpt_path_best = checkpoint_fpath+'/'+'best.pt'
        print("This is the best model\n")
        shutil.copyfile(ckpt_path,
                        ckpt_path_best)


def train(wandb, args, cfg, net_G, net_D):
    train_loader, _, _ = get_data_loader(cfg=cfg, args= args)
    real_label = 1.0
    fake_label = 0.0
    fixed_noise = torch.FloatTensor(np.random.normal(0, 1, (16, cfg.latent_dim))).to(cfg.device)# for visualization
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(net_G.parameters(), lr= cfg.lr, betas = (cfg.beta_1, cfg.beta_2))
    optimizer_d = optim.Adam(net_D.parameters(), lr =cfg.lr, betas = (cfg.beta_1, cfg.beta_2))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
    if cfg.decay_type == "cosine":
        scheduler_g = WarmupCosineSchedule(optimizer_g, warmup_steps=cfg.warmup_steps, t_total=cfg.epochs*len(train_loader))
        scheduler_d = WarmupCosineSchedule(optimizer_d, warmup_steps=cfg.warmup_steps, t_total=cfg.epochs*len(train_loader))
    else:
        scheduler_g = WarmupLinearSchedule(optimizer_g, warmup_steps=cfg.warmup_steps, t_total=cfg.epochs*len(train_loader))
        scheduler_d = WarmupLinearSchedule(optimizer_d, warmup_steps=cfg.warmup_steps, t_total=cfg.epochs*len(train_loader))

    start_epoch = 0
    loss_g = AverageMeter()
    loss_d = AverageMeter()
    D_x_total = AverageMeter()
    D_G_z_total = AverageMeter()

    # We only save the model who uses device "cuda:0"
    # To resume the device for the save model woule be also "cuda:0"
    if args.resume:
        ckpt = load_ckpt(cfg.ckpt_fpath)
        optimizer_g.load_state_dict(ckpt['optimizer_g'])
        optimizer_d.load_state_dict(ckpt['optimizer_d'])
        scheduler_g.load_state_dict(ckpt['scheduler_g'])
        scheduler_d.load_state_dict(ckpt['scheduler_d'])
        start_epoch = ckpt['epoch']+1
    g_step = 0
    for epoch in tqdm(range(start_epoch, cfg.epochs)):
        # Train Descriminator
        net_D.train()
        net_G.train()
        tepoch = tqdm(train_loader,
                disable=args.local_rank not in [-1,0])

        for data in tepoch:
            g_step+=1
            set_grad(net_G,False)
            set_grad(net_D,True)
            #######################################
            # (1) update discriminator (real data)
            #######################################
            optimizer_d.zero_grad()
            inputs = data[0].to(cfg.device, non_blocking=True)
            label = torch.full((inputs.size(0),), real_label, dtype=torch.float, device = cfg.device)
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                outputs = net_D(inputs).view(-1)
                err_real_d = criterion(outputs, label)
            scaler.scale(err_real_d).backward()
            if args.local_rank in [-1,0]:
                D_x = outputs.detach().mean().item()
                D_x_total.update(D_x, inputs.size(0))

            ###############################################
            # (2) update discriminator (fake data)
            ##########################################
            latent_vector = torch.FloatTensor(np.random.normal(0, 1, (inputs.size(0), cfg.latent_dim))).to(cfg.device)
            label.fill_(fake_label)
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                f_img = net_G(latent_vector)
                outputs_fake = net_D(f_img).view(-1)
                err_fake_d = criterion(outputs_fake, label)
            scaler.scale(err_fake_d).backward()
            scaler.step(optimizer_d)
            scaler.update()
            scheduler_d.step()
            if args.local_rank in [-1,0]:
                err_d = err_fake_d + err_real_d
                loss_d.update(err_d.detach().mean().item(), inputs.size(0))

            ####################################################
            # (3) Update Generator
            ####################################################
            set_grad(net_G,True)
            set_grad(net_D,False)
            optimizer_g.zero_grad()
            latent_vector = torch.FloatTensor(np.random.normal(0, 1, (inputs.size(0), cfg.latent_dim))).to(cfg.device)
            label.fill_(real_label)
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                f_img = net_G(latent_vector)
                outputs = net_D(f_img).view(-1)
                err_g = criterion(outputs, label)
            scaler.scale(err_g).backward()
            scaler.step(optimizer_g)
            scaler.update()
            scheduler_g.step()
            if args.local_rank in [-1,0]:
                loss_g.update(err_g.detach().item(),inputs.size(0))
                D_G_z = outputs.mean().item()
                D_G_z_total.update(D_G_z,inputs.size(0))
            exp_mov_avg(net_Gs, net_G, global_step = g_step)
            if args.local_rank in [-1,0]:
                tepoch.set_description(f'Epoch {epoch}: loss_D: {loss_d.avg:2.4f}, Loss_G: {loss_g.avg:2.4f}, lr : {scheduler_d.get_lr()[0]:.2E}')

        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                fake_sample = net_G(fixed_noise).detach().cpu()


        model_to_save_g = net_G.module if hasattr(net_G, 'module') else net_G
        model_to_save_d = net_D.module if hasattr(net_D, 'module') else net_D
        ckpt = {
                'model_g': model_to_save_g.state_dict(),
                'model_d': model_to_save_d.state_dict(),
                'optimizer_g': optimizer_g.state_dict(),
                'optimizer_d': optimizer_d.state_dict(),
                'scheduler_g': scheduler_g.state_dict(),
                'scheduler_d': scheduler_d.state_dict(),
                'epoch': epoch
                }
        save_ckpt(checkpoint_fpath=cfg.ckpt_fpath, checkpoint=ckpt, is_best=True)
        if cfg.wandb.active:
            wandb.log({
                "loss_D": loss_d.avg,
                "loss_G": loss_g.avg,
                "D(x)" : D_x_total.avg,
                "D(G(z))": D_G_z_total.avg,
                },
                step = epoch )
            img = wandb.Image( vutils.make_grid(fake_sample, padding=2,normalize=True).numpy().transpose(1,2,0), caption= f"Generated @ {epoch}")
            wandb.log({"Examples": img}, step=epoch)
        if args.local_rank in [-1,0]:
            print("-"*75+ "\n")
            print(f"| {epoch}-th epoch, D(x): {D_x_total.avg:2.2f}, D(G(z)): {D_G_z_total.avg:2.2f}\n")
            print("-"*75+ "\n")
        if args.multigpu:
            torch.distributed.barrier()

def test(wandb, args, cfg, net_G):
    train_loader, _, _ = get_data_loader(cfg=cfg, args= args)
    fixed_noise = torch.FloatTensor(np.random.normal(0, 1, (64, cfg.latent_dim))).to(cfg.device)# for visualization
    net_G.eval()
    if args.local_rank in [-1,0]:
        with torch.no_grad():
            bat = next(iter( train_loader))
            img_real = wandb.Image(
                    vutils.make_grid(bat[0].to(cfg.device)[:64], padding=5, normalize=True),
                    caption = "Real Image"
                    )
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                img = net_G(fixed_noise).detach().cpu()
            img_fake = wandb.Image(
                    vutils.make_grid(img[:64], padding=5, normalize=True),
                    caption = "fake image"
                    )
            wandb.log({"img_real": img_real, "img_fake": img_fake})



def main():
    args = parse_args()
    cfg = edict(yaml.safe_load(open(args.config)))
    if args.multigpu:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        if int(os.environ["RANK"]) !=0:
            cfg.wandb.active=False
    else:
        args.local_rank = -1

    if args.local_rank in [-1,0] and cfg.wandb.active:
        wandb.init(project = cfg.wandb.project,
                   entity = cfg.wandb.id,
                   config = dict(cfg),
                   name = f'{cfg.wandb.name}_lr:{cfg.lr}_fp16:{cfg.use_amp}',
                   group = cfg.dataset
                   )
    if args.local_rank in [-1,0]:
        print(cfg)
    # set cuda flag
    if not torch.cuda.is_available():
        print("WARNING: You have a CUDA device, so you should probably enable CUDA")
    # Set Automatic Mixed Precision
    # We need to use seeds to make sure that model initialization same
    set_random_seeds(random_seed = cfg.random_seed)
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    if args.multigpu: 
        torch.distributed.init_process_group(backend="nccl")
        # torch.distributed.init_process_group(backend="gloo")
        cfg.device = torch.device("cuda:{}".format(args.local_rank))
    else:
        cfg.device = torch.device("cuda:0")

    # torch.distributed.init_process_group(backend="gloo")

    # Encapsulate the model on the GPU assigned to the current process
    net_G = Generator(grayscale=cfg.grayscale, initialize_size=cfg.initialize_size, blocks= cfg.blocks)
    net_D = Discriminator(grayscale=cfg.grayscale, patch_size = cfg.patch_size, blocks=cfg.blocks, extend_size=cfg.extend_size)
    # if custom_pre-trained model : model.load_from(np.load(<path>))
    net_G = net_G.to(cfg.device)
    net_D = net_D.to(cfg.device)
    if args.resume or args.test:
        ckpt = load_ckpt(cfg.ckpt_fpath, is_best = args.test)
        net_G.load_state_dict(ckpt['model_g'])
        net_D.load_state_dict(ckpt['model_d'])

    #pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(f'Number of trainable parameter is {pytorch_total_params:.2E}')
    if args.multigpu:
        net_G = torch.nn.parallel.DistributedDataParallel(
                net_G, 
                device_ids=[args.local_rank], 
                output_device=args.local_rank, 
                find_unused_parameters=True)
        net_D = torch.nn.parallel.DistributedDataParallel(
                net_D, 
                device_ids=[args.local_rank], 
                output_device=args.local_rank,
                find_unused_parameters=True)



    if args.test:
        test(wandb, args, cfg, net_G)
    else:
        train(wandb, args, cfg, net_G, net_D)


if __name__ == '__main__':
    main()
