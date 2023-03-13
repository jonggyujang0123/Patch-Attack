"""========Default import===="""
import argparse
import os
from utils.base_utils import set_random_seeds, get_accuracy, AverageMeter, WarmupCosineSchedule, WarmupLinearSchedule
import torch.nn as nn
import torch
import wandb
import os
import torch.optim as optim
import shutil
from tqdm import tqdm
from easydict import EasyDict as edict
import yaml
from utils.patch_utils import patch_util
import numpy as np
from einops import rearrange
""" ==========END================"""

""" =========Configurable ======="""
from models.resnet18 import R_18_MNIST as resnet
from models.miner import ReparameterizedMVN
from models.miner_linear import ReparameterizedMVN_Linear, ReparameterizedGMM_Linear, gaussian_logp
from models.CVAE_coder import CVAE
from models.CGAN import Generator


#os.environ["WANDB_SILENT"] = 'true'

""" ===========END=========== """

def parse_args():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config",
            type=str, 
            help="Configuration file in configs.")
    parser.add_argument("--multigpu",
            type=bool, 
            help="Local rank. Necessary for using the torch.distributed.launch utility.",
            default= False)
    parser.add_argument("--resume",
            type=int,
            default= 0,
            help="Resume the last training from saved checkpoint.")
    parser.add_argument("--test", 
            type=int,
            default = 0,
            help="if test, choose True.")
    parser.add_argument("--fixed-id", 
            type=int,
            default = 0,
            help="target index.")
    args = parser.parse_args()
    return args


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

class LabelSMoothingLoss(nn.Module):
    def __init__(
            self,
            n_classes,
            smoothing=0.0,
            dim=-1):
        super(LabelSMoothingLoss, self).__init__()
        self.confidence = 1 - smoothing
        self.smoothing = smoothing
        self.n_class = n_classes
        self.dim=dim

    def forward(self, lsm, target):
        true_dist= torch.zeros_like(lsm)
        true_dist.fill_(self.smoothing / (self.n_class - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum( - true_dist * lsm,  dim = self.dim))




def generate_img(
        model,
        cfg,
        w,
        is_gif = False,
        is_freeze = False):
    patch_u = patch_util(
            img_size = cfg.img_size,
            patch_size = cfg.patch_size,
            patch_margin = cfg.patch_margin,
            device = cfg.device)
    inputs = torch.zeros(w.shape[0], cfg.num_channel, cfg.img_size, cfg.img_size).to(cfg.device) 
    samples = list()
#    print(model.last[0].bias)
#    print(model.last[0].weight)
    for pat_ind_const in range(patch_u.num_patches):
        pat_ind = torch.tensor(np.ones(shape=(w.shape[0],), dtype=int)).to(w.device) * pat_ind_const
        if is_freeze:
            with torch.no_grad():
                _, cond_img, _,_ = patch_u.get_patch(inputs, pat_ind = pat_ind)
        else:
            _, cond_img, _, _ = patch_u.get_patch(inputs, pat_ind = pat_ind)
#        new_patch = model.decode(w[:, pat_ind_const,:], cond_img)
        new_patch = model(w[:, pat_ind_const,:], cond_img)
#        print(new_patch[0,:])
        inputs = patch_u.concat_patch(inputs, new_patch, pat_ind)
        samples.append(inputs)
    return samples if is_gif else inputs




def train(wandb, args, cfg, classifier, generator, miner):
    fixed_z = torch.randn(
            cfg.test_batch_size, 
            cfg.n_patch, 
            cfg.latent_size
            ).to(cfg.device)
    attack_criterion = LabelSMoothingLoss(
            cfg.num_classes, smoothing=cfg.attack_labelsmooth)
    ## Setting 
    optimizer = optim.SGD(miner.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    if cfg.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=cfg.warmup_steps, t_total=cfg.epochs)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=cfg.warmup_steps, t_total=cfg.epochs)
    start_epoch =0
    AvgLoss = AverageMeter()
    AvgAcc = AverageMeter()
    best_acc = 0.0
    # We only save the model who uses device "cuda:0"
    # To resume the device for the save model woule be also "cuda:0"
    if args.resume:
        ckpt = load_ckpt(cfg.ckpt_fpath)
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']+1

    pbar = tqdm(
            range(start_epoch,cfg.epochs),
            disable = args.local_rank not in [-1,0]
            )
    # Prepare dataset and dataloader
    for epoch in pbar:
#        if args.local_rank not in [-1,1]:
#            print("Local Rank: {}, Epoch: {}, Training ...".format(args.local_rank, epoch))
        miner.train()
        optimizer.zero_grad()
        z = torch.randn(
                cfg.train_batch_size, 
                cfg.n_patch, 
                cfg.latent_size
                ).to(cfg.device)
        w = miner(z) 
        fake = generate_img(
                generator,
                cfg,
                w,
                is_freeze= False)#True if epoch % cfg.freeze_every == 0 else False)

        # Compute Loss
        ## Classifier Loss
        sftm = nn.Softmax(dim = -1)
#        LSM = torch.log(1e-6 + sftm(classifier(fake)))
#        LSM = torch.log(1e-6 + sftm(classifier(fake + 0.6*(cfg.epochs - epoch) / cfg.epochs * torch.randn_like(fake, device=fake.device))))
#        LSM = torch.log(1e-6 + sftm(classifier(fake + (0.7 + 0.3*(cfg.epochs - epoch) / cfg.epochs) * torch.randn_like(fake, device=fake.device))))
        LSM = torch.log(1e-6 + sftm(classifier(fake + (0.2 + 0.1*(cfg.epochs - epoch) / cfg.epochs) * torch.randn_like(fake, device=fake.device))))
#        LSM = torch.log(1e-6 + sftm(classifier(fake + (0.0 + 0.0*(cfg.epochs - epoch) / cfg.epochs) * torch.randn_like(fake, device=fake.device))))
        fake_y = args.fixed_id * \
                torch.ones(cfg.train_batch_size).to(cfg.device).long()
        loss_attack = attack_criterion(LSM, fake_y)
#        train_acc = (LSM.max(1)[1] == fake_y).float().mean().item()
        train_acc = LSM[:,args.fixed_id].exp().mean().item()
        ## Entropy Loss
        if cfg.lambda_miner_entropy > 0:
            loss_miner_entropy = miner.entropy()
        else:
            loss_miner_entropy = 0
        ## KL Diverg Loss
        if epoch % cfg.kl_every ==0 and cfg.lambda_kl > 0:
            if False:
                samples = miner(torch.randn(
                    cfg.train_batch_size,
                    cfg.n_patch * cfg.latent_size).to(cfg.device))
                loss_kl = torch.mean(miner.logp(samples) - gaussian_logp(torch.zeros_like(samples), torch.zeros_like(samples), samples).sum(-1).sum(-1))
            else:
                samples = torch.randn(
                    cfg.train_batch_size, 
                    cfg.n_patch, 
                    cfg.latent_size
                    ).to(cfg.device)
#                L = torch.tril(miner.L)
                C = miner.L @ miner.L.T 
    #            C = torch.einsum('pkl,plj->pkj', miner.L ,miner.L.transpose(1,2))
                loss_kl = (1/2) * (
                                torch.norm(miner.m.view([-1, cfg.n_patch * cfg.latent_size]), p=2, dim=[-1]).pow(2).mean() - \
                                torch.logdet(C) + \
                                torch.trace(C)
                                )
        else:
            loss_kl = 0.0
        loss = cfg.lambda_attack * loss_attack + \
                cfg.lambda_miner_entropy * loss_miner_entropy + \
                cfg.lambda_kl * loss_kl
        #print(f'loss kl : {loss_kl}, loss_attack : {loss_attack}')
        loss.backward()
        optimizer.step()
        scheduler.step() #optimizer.step()
#        print(LSM)

        AvgLoss.update(loss.item(), n =z.shape[0])
        AvgAcc.update(train_acc, n =z.shape[0])

        if args.local_rank in [-1,0]:
            pbar.set_description(f'Epoch {epoch}: train_loss: {AvgLoss.avg:2.2f}, lr : {scheduler.get_lr()[0]:.2E}, mean: {w.abs().mean().item():.4f}, tr_acc : {AvgAcc.avg:.2f}')

        if epoch % cfg.interval_val == 0:
            if args.local_rank in [-1, 0]:
                val_acc = evaluate(
                        wandb = wandb,
                        args =args,
                        cfg = cfg,
                        classifier = classifier,
                        generator = generator,
                        miner = miner,
                        epoch = epoch,
                        fixed_noise = fixed_z)
                model_to_save = miner.module if hasattr(miner, 'module') else miner
                ckpt = {
                        'model': model_to_save.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch
                        }
                save_ckpt(checkpoint_fpath=cfg.ckpt_fpath, checkpoint=ckpt)
                if best_acc < val_acc:
                    save_ckpt(checkpoint_fpath=cfg.ckpt_fpath, checkpoint=ckpt, is_best=True)
                    best_acc = val_acc
                if cfg.wandb.active:
                    wandb.log({
                        "loss": AvgLoss.val,
                        "val_acc" : val_acc
                        })
                print("-"*75+ "\n")
                print(f"| {epoch}-th epoch, training loss is {AvgLoss.avg:.4f}, Val Accuracy is {val_acc*100:2.2f}  \n")
                print("-"*75+ "\n")
            AvgLoss.reset()
            AvgAcc.reset()
    pbar.close()


def evaluate(wandb, args, cfg, classifier, generator, miner, epoch = 0, fixed_noise = None):
    if fixed_noise == None:
        z = torch.randn(
                cfg.test_batch_size, 
                cfg.n_patch, 
                cfg.latent_size
                ).to(cfg.device)
    else:
        z = fixed_noise
    val_acc = 0
    time_step = 1#16
    if args.local_rank in [-1,0]:
        miner.eval()
        w = miner(z*0) 
        fake = generate_img(
                generator,
                cfg,
                w,
                is_gif = True)
        pred = classifier(fake[-1])
        fake_y = args.fixed_id * \
                torch.ones(cfg.test_batch_size).to(cfg.device)
        val_acc = (pred.max(1)[1] == fake_y).float().mean().item()

        gif = rearrange(torch.cat(fake),
                '(t b1 b2) c h w -> t (b1 h) (b2 w) c', b1 = cfg.test_batch_size // 16, b2 = 16)
        gif = rearrange(gif,
                '(t1 t2) h w c -> t1 t2 h w c', 
                t1 = time_step,
                )[:, -1, :, :, :].permute(0, 3, 1, 2)
        gif = 255 * gif.cpu().detach().numpy().astype(np.float64)
        gif = wandb.Video(
                gif.astype(np.uint8),
                fps = 4,
                format=  'gif'
                )
        wandb.log({f'reconstruct_{args.fixed_id}':gif}, step = epoch)

    return val_acc

def main():
    args = parse_args()
    cfg = edict(yaml.safe_load(open(args.config)))
    cfg.n_patch =(cfg.img_size // cfg.patch_size)**2
    if args.multigpu:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        if int(os.environ["RANK"]) !=0:
            cfg.wandb.active=False
    else:
        args.local_rank = -1

    if args.local_rank in [-1,0] and args.test==False and cfg.wandb.active:
        wandb.init(project = cfg.wandb.project,
                   entity = cfg.wandb.id,
                   config = dict(cfg),
                   name = f'{cfg.wandb.name}_lr:{cfg.lr}',
                   group = f'{cfg.dataset}_P{cfg.patch_size}_M{cfg.patch_margin}_Tar{args.fixed_id}'
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
#    miner = ReparameterizedGMM_Linear(
#            n_patch = cfg.n_patch,
#            n_z = cfg.latent_size,
#            ).to(cfg.device)
    miner = ReparameterizedMVN_Linear(
            n_patch = cfg.n_patch,
            n_z = cfg.latent_size,
            ).to(cfg.device)

    classifier = resnet(
            num_classes= cfg.num_classes,
            grayscale= cfg.grayscale
            ).to(cfg.device)
    generator = Generator(
            patch_size = cfg.patch_size,
            patch_margin = cfg.patch_margin,
            latent_size = cfg.latent_size,
            n_gf = cfg.n_gf,
            n_c = 1
            ).to(cfg.device)

#    generator = CVAE(
#            img_size = cfg.img_size,
#            latent_size = cfg.latent_size,
#            patch_size = cfg.patch_size,
#            patch_margin = cfg.patch_margin,
#            emb_size = cfg.emb_size,
#            pos_emb_size = cfg.pos_emb_size,
#            grayscale = cfg.grayscale,
#            training = False
#            ).to(cfg.device)

    # Load target classifier 
    if args.multigpu:
        generator = torch.nn.parallel.DistributedDataParallel(generator, device_ids=[args.local_rank], output_device=args.local_rank)
        classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[args.local_rank], output_device=args.local_rank)
        miner = torch.nn.parallel.DistributedDataParallel(miner, device_ids=[args.local_rank], output_device=args.local_rank)
    if os.path.exists(cfg.ckpt_fpath_class):
        ckpt = load_ckpt(cfg.ckpt_fpath_class, is_best = True)
        classifier.load_state_dict(ckpt['model'])
        classifier.eval()
        print(f'{cfg.ckpt_fpath_class} model is loaded!')
    else:
        raise Exception('there is no generative checkpoint')
    # Load common generative model
    if os.path.exists(cfg.ckpt_fpath_generator):
        ckpt = load_ckpt(cfg.ckpt_fpath_generator, is_best = True)
        print(f'{cfg.ckpt_fpath_generator} model is loaded!')
        generator.load_state_dict(ckpt['model_g'])
#        generator.load_state_dict(ckpt['model'])
        generator.eval()
#        print(generator.last[0].bias)
    else:
        raise Exception('there is no generative checkpoint')
    # Load Miner if resume or test
    if args.resume or args.test:
        ckpt = load_ckpt(cfg.ckpt_fpath, is_best = args.test)
        miner.load_state_dict(ckpt['model'])




    if args.test:
        evaluate(wandb, args, cfg, classifier, generator, miner)
    else:
        train(wandb, args, cfg, classifier, generator, miner)


if __name__ == '__main__':
    main()
