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
from models.CGAN import Generator, Discriminator
from utils.patch_utils import patch_util
import torch.nn.functional as F 
from einops import rearrange


""" ==========END================"""

""" =========Configurable ======="""
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

def get_data_loader(cfg, args, dataset):
    if dataset in ['cifar100', 'cifar10']:
        from datasets.cifar import get_loader_cifar as get_loader
    if dataset in ['mnist']:
        from datasets.mnist import get_loader_mnist as get_loader
    if dataset in ['emnist']:
        from datasets.emnist import get_loader_emnist as get_loader
    if dataset in ['fashion']:
        from datasets.fashion_mnist import get_loader_fashion_mnist as get_loader
    return get_loader(cfg, args)


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


def train(wandb, args, cfg, G, D):
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(G.parameters(), lr =cfg.lr, betas = (cfg.beta_1, cfg.beta_2))
    optimizer_d = optim.Adam(D.parameters(), lr =cfg.lr, betas = (cfg.beta_1, cfg.beta_2))
    train_loader, _, _ = get_data_loader(cfg=cfg, args= args, dataset=cfg.dataset)
    patch_u = patch_util(img_size= cfg.img_size,
            patch_size = cfg.patch_size,
            patch_margin = cfg.patch_margin,
            device = cfg.device)
    fixed_noise = torch.FloatTensor(np.random.normal(0, 1, (cfg.test_batch_size, patch_u.num_patches, cfg.latent_size))).to(cfg.device)# for visualization
    fixed_noise[0,:,:] = 0.0

    if cfg.decay_type == "cosine":
        scheduler_g = WarmupCosineSchedule(optimizer_g, warmup_steps=cfg.warmup_steps, t_total=cfg.epochs*len(train_loader))
        scheduler_d = WarmupCosineSchedule(optimizer_d, warmup_steps=cfg.warmup_steps, t_total=cfg.epochs*len(train_loader))
    else:
        scheduler_g = WarmupLinearSchedule(optimizer_g, warmup_steps=cfg.warmup_steps, t_total=cfg.epochs*len(train_loader))
        scheduler_d = WarmupLinearSchedule(optimizer_d, warmup_steps=cfg.warmup_steps, t_total=cfg.epochs*len(train_loader))

    start_epoch = 0
    real_label = 1.0
    fake_label = 0.0 
    # We only save the model who uses device "cuda:0"
    # To resume the device for the save model woule be also "cuda:0"
    if args.resume:
        ckpt = load_ckpt(cfg.ckpt_fpath)
        optimizer_g.load_state_dict(ckpt['optimizer_g'])
        optimizer_d.load_state_dict(ckpt['optimizer_d'])
        scheduler_g.load_state_dict(ckpt['scheduler_g'])
        scheduler_d.load_state_dict(ckpt['scheduler_d'])
        start_epoch = ckpt['epoch']+1
    Loss_g = AverageMeter()
    Loss_d = AverageMeter()
    D_x_total = AverageMeter()
    D_G_z_total = AverageMeter()
    for epoch in tqdm(range(start_epoch, cfg.epochs)):
        # Train Descriminator
        G.train()
        D.train()
        tepoch = tqdm(train_loader,
                disable=args.local_rank not in [-1,0])

        for data in tepoch:
            ############################################
            # (1) Update Discriminator (Real Data)
            ############################################
            optimizer_d.zero_grad()
            inputs = data[0].to(cfg.device, non_blocking=True)
            label = torch.full((inputs.size(0),), real_label, dtype=torch.float, device = cfg.device)
            patch, cond_img, _, origin = patch_u.get_patch(inputs)
            outputs = D(origin).view(-1)
            err_real_d = criterion(outputs, label)
            if args.local_rank in [-1,0]:
                D_x = outputs.detach().mean().item()
                D_x_total.update(D_x, inputs.size(0))
            err_real_d.backward()
            optimizer_d.step()

            #############################################
            # (2) Update Discriminator (Fake data)
            ################################################

            optimizer_d.zero_grad()
            latent_vector = torch.randn(inputs.size(0), cfg.latent_size, 1, 1, device= cfg.device)
            label.fill_(fake_label)
            fake = G(latent_vector, cond_img)
            fake = patch_u.concat_extended_patch(fake, cond_img)
            outputs = D(fake.detach()).view(-1)
            err_fake_d = criterion(outputs, label)
            err_fake_d.backward()
            optimizer_d.step()
            if args.local_rank in [-1, 0]:
                err_d = err_fake_d + err_real_d
                Loss_d.update(err_d.detach().mean().item(), inputs.size(0))

            #################################################
            # (3) Update Generator
            ####################################################

            optimizer_g.zero_grad()
            label.fill_(real_label)
            outputs = D(fake).view(-1)
            err_g = criterion(outputs, label)
            err_g.backward()
            optimizer_g.step()

            scheduler_d.step()
            scheduler_g.step()

            if args.local_rank in [-1, 0]:
                Loss_g.update(err_g.detach().item(), inputs.size(0))
                D_G_z = outputs.mean().item()
                D_G_z_total.update(D_G_z, inputs.size(0))
                tepoch.set_description(f'Epoch {epoch}: Loss_D : {Loss_d.avg:2.4f}, Loss_G: {Loss_g.avg:2.4f}, lr is {scheduler_d.get_lr()[0]:.2E}')

        ###################################################
        # (4) Output of the training
        ####################################################
        if args.local_rank in [-1,0]:
            model_to_save_g = G.module if hasattr(G, 'module') else G
            model_to_save_d = D.module if hasattr(D, 'module') else D
            ckpt = {
                    'model_g' : model_to_save_g.state_dict(),
                    'model_d' : model_to_save_d.state_dict(),
                    'optimizer_g' : optimizer_g.state_dict(),
                    'optimizer_d' : optimizer_d.state_dict(),
                    'scheduler_g' : scheduler_g.state_dict(),
                    'scheduler_d' : scheduler_d.state_dict()
                    }
            save_ckpt(checkpoint_fpath = cfg.ckpt_fpath, checkpoint = ckpt, is_best=True)
            if cfg.wandb.active:
                wandb.log({
                    "loss_D" : Loss_d.avg,
                    "Loss_G" : Loss_g.avg,
                    "D(x)" : D_x_total.avg,
                    "D(G(z))" : D_G_z_total.avg
                    },
                    step = epoch)
        Loss_g.reset()
        Loss_d.reset()
        D_G_z_total.reset()
        D_x_total.reset()
        if epoch % cfg.eval_every == 0:
            test(wandb, args, cfg, G, fixed_noise=fixed_noise, epoch = epoch)
#            for _, item in cfg.dataset_2.items():
#                test(wandb, args, cfg, model, fixed_noise=fixed_noise, epoch = epoch, dataset = item)
        if args.multigpu:
            torch.distributed.barrier()

def test(wandb, args, cfg, G, fixed_noise = None, epoch = 0):
    G.eval()
    patch_u = patch_util(img_size= cfg.img_size,
            patch_size = cfg.patch_size,
            patch_margin = cfg.patch_margin,
            device = cfg.device)

    time_step = patch_u.num_patches
    if fixed_noise == None:
        fixed_noise = torch.FloatTensor(np.random.normal(0, 1, (cfg.test_batch_size, patch_u.num_patches, cfg.latent_size))).to(cfg.device)# for visualization
        fixed_noise[0,:,:] = 0.0
    _, _, test_loader = get_data_loader(cfg=cfg, args= args, dataset=cfg.dataset)
    # Free drawing
    bat = next(iter(test_loader))
    inputs = 0* bat[0].to(cfg.device, non_blocking=True)
    #pat_ind = torch.tensor(np.zeros(shape=(cfg.test_batch_size,), dtype=int)).to(cfg.device)
    samples = list()
    if args.local_rank in [-1,0]:
        with torch.no_grad():
            for pat_ind_const in range(patch_u.num_patches):
                pat_ind = torch.tensor(np.ones(shape=(cfg.test_batch_size,), dtype=int)).to(cfg.device) * pat_ind_const
                _, cond_img, pat_ind,_ = patch_u.get_patch(inputs, pat_ind = pat_ind)
                new_patch = G(fixed_noise[:, pat_ind_const, :], cond_img)
#                new_patch = G(fixed_noise[:, pat_ind_const, :]*0, cond_img*0)
#                print(new_patch[0,:])
#                print(G.last[0].weight)
#                print(G.last[0].bias)
                inputs = patch_u.concat_patch(inputs, new_patch, pat_ind)
                samples.append(inputs)
#                if pat_ind_const == 0:
#                    print(new_patch[0,:])
        gif = rearrange(torch.cat(samples),
                '(t b1 b2) c h w -> t (b1 h) (b2 w) c', b1=8, b2=8
                ).cpu().numpy() 
        gif = 255 * gif.astype(np.float64)
        gif = rearrange(gif, '(t1 t2) h w c -> t1 t2 h w c', t2 = time_step)[:,-1,:,:,:].transpose(0, 3, 1, 2)
        gif = wandb.Video(
                gif.astype(np.uint8),
                fps = 4,
                format = "gif"
                )
#        gif = wandb.Image(gif.astype(np.uint8)[0,:,:,:])
        if cfg.wandb.active:
            wandb.log({
                "img": gif},
                step= epoch)

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
                   name = f'{cfg.wandb.name}_lr:{cfg.lr}',
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
    G = Generator(
            patch_size=cfg.patch_size,
            patch_margin = cfg.patch_margin,
            latent_size= cfg.latent_size,
            n_gf = cfg.n_gf,
            n_c = 1 if cfg.grayscale else 3
            )
    D = Discriminator(
            patch_size= cfg.patch_size,
            patch_margin = cfg.patch_margin,
            n_df = cfg.n_df,
            n_c = 1 if cfg.grayscale else 3
            )
    # if custom_pre-trained model : model.load_from(np.load(<path>))
    G = G.to(cfg.device)
    D = D.to(cfg.device)
    if args.resume or args.test:
        ckpt = load_ckpt(cfg.ckpt_fpath, is_best = args.test)
        G.load_state_dict(ckpt['model_g'])
        D.load_state_dict(ckpt['model_d'])

    #pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(f'Number of trainable parameter is {pytorch_total_params:.2E}')
    if args.multigpu:
        G = torch.nn.parallel.DistributedDataParallel(
                G, 
                device_ids=[args.local_rank], 
                output_device=args.local_rank, 
                find_unused_parameters=True)
        D = torch.nn.parallel.DistributedDataParallel(
                D, 
                device_ids=[args.local_rank], 
                output_device=args.local_rank, 
                find_unused_parameters=True)


    if args.test:
        test(wandb, args, cfg, G)
    else:
        train(wandb, args, cfg, G, D)


if __name__ == '__main__':
    main()
