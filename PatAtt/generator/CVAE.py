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
from models.CVAE_coder import CVAE
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



def loss_function(recon_x, x, mu ,log_var, beta=0.01, epsilon = 0.01):
    x = torch.clamp(x, 1e-8, 1-1e-8)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1+log_var - mu.pow(2) - log_var.exp())
    return BCE + beta *  (KLD - epsilon)

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


def train(wandb, args, cfg, model):
    optimizer = optim.Adam(model.parameters(), lr =cfg.lr, betas = (cfg.beta_1, cfg.beta_2))
    train_loader, _, _ = get_data_loader(cfg=cfg, args= args, dataset=cfg.dataset)
    fixed_noise = torch.FloatTensor(np.random.normal(0, 1, (cfg.test_batch_size, 64, cfg.latent_size))).to(cfg.device)# for visualization
    patch_u = patch_util(img_size= cfg.img_size,
            patch_size = cfg.patch_size,
            patch_margin = cfg.patch_margin,
            device = cfg.device)

    if cfg.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=cfg.warmup_steps, t_total=cfg.epochs*len(train_loader))
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=cfg.warmup_steps, t_total=cfg.epochs*len(train_loader))

    start_epoch = 0
    best_loss = 999 
    # We only save the model who uses device "cuda:0"
    # To resume the device for the save model woule be also "cuda:0"
    if args.resume:
        ckpt = load_ckpt(cfg.ckpt_fpath)
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']+1
    Loss = AverageMeter()
    for epoch in tqdm(range(start_epoch, cfg.epochs)):
        # Train Descriminator
        model.train()
        tepoch = tqdm(train_loader,
                disable=args.local_rank not in [-1,0])

        for data in tepoch:
            optimizer.zero_grad()
            inputs = data[0].to(cfg.device, non_blocking=True)
            patch, cond_img, pat_ind = patch_u.get_patch(inputs)
            recon_batch, mu, log_var = model.forward(patch, cond_img, pat_ind)
            loss = loss_function(recon_batch, patch, mu, log_var)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if args.local_rank in [-1,0]:
                Loss.update(loss.mean().item() / patch.size(0) ,patch.size(0))
                tepoch.set_description(f'Epoch {epoch}: loss is {Loss.avg}, lr is {scheduler.get_lr()[0]:.2E}')
        if epoch % cfg.eval_every == 0:
            test(wandb, args, cfg, model, fixed_noise=fixed_noise, epoch = epoch)
            for _, item in cfg.dataset_2.items():
                test(wandb, args, cfg, model, fixed_noise=fixed_noise, epoch = epoch, dataset = item)
        if args.local_rank in [-1,0]:
            model_to_save = model.module if hasattr(model, 'module') else model
            ckpt = {
                    'model': model_to_save.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch
                    }
            if best_loss > Loss.avg:
                save_ckpt(checkpoint_fpath = cfg.ckpt_fpath, checkpoint= ckpt, is_best=True)
                best_loss = Loss.avg
            else:
                save_ckpt(checkpoint_fpath = cfg.ckpt_fpath, checkpoint= ckpt)
            if cfg.wandb.active:
                wandb.log({
                    "loss": Loss.avg,
                    }, step= epoch)
        if args.multigpu:
            torch.distributed.barrier()

def test(wandb, args, cfg, model, fixed_noise = None, epoch = 0, dataset=None):
    model.eval()
    patch_u = patch_util(img_size= cfg.img_size,
            patch_size = cfg.patch_size,
            patch_margin = cfg.patch_margin,
            device = cfg.device)

    time_step = patch_u.num_patches//16
    if fixed_noise == None:
        fixed_noise = torch.FloatTensor(np.random.normal(0, 1, (cfg.test_batch_size, 64, cfg.latent_size))).to(cfg.device)# for visualization
    if dataset == None:
        _, _, test_loader = get_data_loader(cfg=cfg, args= args, dataset=cfg.dataset)
        # Free drawing
        bat = next(iter(test_loader))
        inputs = 0* bat[0].to(cfg.device, non_blocking=True)
        pat_ind = torch.tensor(np.zeros(shape=(cfg.test_batch_size,), dtype=int)).to(cfg.device)
        samples = list()
        if args.local_rank in [-1,0]:
            with torch.no_grad():
                for pat_ind_const in range(patch_u.num_patches):
                    pat_ind = torch.tensor(np.ones(shape=(cfg.test_batch_size,), dtype=int)).to(cfg.device) * pat_ind_const
                    _, cond_img, pat_ind = patch_u.get_patch(inputs, pat_ind = pat_ind)
                    new_patch = model.decode(fixed_noise[:, pat_ind_const, :], cond_img, pat_ind)
                    inputs = patch_u.concat_patch(inputs, new_patch, pat_ind)
#                    samples.append(inputs)
            gif = rearrange(torch.cat(samples),
                    '(t b1 b2) c h w -> t (b1 h) (b2 w) c', b1=4, b2=4
                    ).cpu().numpy() 
            gif = 255 * gif.astype(np.float64)
            gif = rearrange(gif, '(t1 t2) h w c -> t1 t2 h w c', t2 = time_step)[:,-1,:,:,:].transpose(0, 3, 1, 2)
            gif = wandb.Video(
                    gif.astype(np.uint8),
                    fps = 4,
                    format = "gif"
                    )
            if cfg.wandb.active:
                wandb.log({
                    f"gif_free_draw": gif},
                    step= epoch)
    else: 
        # Guided drawing
        _, _, test_loader = get_data_loader(cfg=cfg, args= args, dataset=dataset)
        bat = next(iter(test_loader))
        inputs = bat[0].to(cfg.device, non_blocking=True)
        samples = list()
        if args.local_rank in [-1,0]:
            with torch.no_grad():
                for pat_ind_const in range(patch_u.num_patches):
                    pat_ind = torch.tensor(np.ones(shape=(cfg.test_batch_size,), dtype=int)).to(cfg.device) * pat_ind_const
                    patch, cond_img, pat_ind = patch_u.get_patch(inputs, pat_ind = pat_ind)
                    embedding, _ = model.encode(patch, cond_img, pat_ind)
                    new_patch = model.decode(embedding, cond_img, pat_ind)
                    inputs = patch_u.concat_patch(inputs, new_patch, pat_ind)
                    samples.append(inputs)
            gif = rearrange(torch.cat(samples),
                    '(t b1 b2 ) c h w -> t (b1 h) (b2 w) c', b1=4, b2=4
                    ).cpu().numpy() 
            gif = 255 * gif.astype(np.float64)
            gif = rearrange(gif, '(t1 t2) h w c -> t1 t2 h w c', t2 = time_step)[:,-1,:,:,:].transpose(0, 3, 1, 2)
            gif = wandb.Video(
                    gif.astype(np.uint8),
                    fps = 4,
                    format = "gif"
                    )
    
            img_real = wandb.Image(
                    vutils.make_grid(bat[0].to(cfg.device)[:cfg.test_batch_size], padding=5, normalize=True),
                    caption = "Real Image"
                    )
            if cfg.wandb.active:
                wandb.log({f"gif_guided{dataset}": gif},step=epoch)
                #wandb.log({f"guide image{dataset}": img_real},step=epoch)
                




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
    model = CVAE(
            img_size = cfg.img_size,
            latent_size = cfg.latent_size,
            patch_size = cfg.patch_size,
            patch_margin= cfg.patch_margin,
            emb_size = cfg.emb_size,
            pos_emb_size = cfg.pos_emb_size,
            grayscale=cfg.grayscale,
            training = not args.test
            )
    # if custom_pre-trained model : model.load_from(np.load(<path>))
    model = model.to(cfg.device)
    if args.resume or args.test:
        ckpt = load_ckpt(cfg.ckpt_fpath, is_best = args.test)
        model.load_state_dict(ckpt['model'])

    #pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(f'Number of trainable parameter is {pytorch_total_params:.2E}')
    if args.multigpu:
        model = torch.nn.parallel.DistributedDataParallel(
                model, 
                device_ids=[args.local_rank], 
                output_device=args.local_rank, 
                find_unused_parameters=True)


    if args.test:
        test(wandb, args, cfg, model)
    else:
        train(wandb, args, cfg, model)


if __name__ == '__main__':
    main()
