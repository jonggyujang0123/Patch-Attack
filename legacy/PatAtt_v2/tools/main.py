"""Import default libraries"""
import os
from utils.base_utils import set_random_seeds, get_accuracy, AverageMeter, WarmupCosineSchedule, WarmupLinearSchedule, load_ckpt, save_ckpt
from utils.parameters import para_config
import torch.nn as nn
import torch
import wandb
import os
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from einops import rearrange
from torchvision.transforms import Pad 

"""import target classifier, generator, and discriminator """
from models.resnet18 import R_18_MNIST as resnet
from models.CGAN import Generator, Discriminator
#from models.miner import ReparameterizedMVN
#from models.miner_linear import ReparameterizedMVN_Linear, ReparameterizedGMM_Linear, gaussian_logp
#from models.CVAE_coder import CVAE

#os.environ["WANDB_SILENT"] = 'true'

def get_data_loader(args):
    if args.dataset in ['cifar100', 'cifar10']:
        from datasets.cifar import get_loader_cifar as get_loader
    if args.dataset in ['mnist']:
        from datasets.mnist import get_loader_mnist as get_loader
    if args.dataset in ['emnist']:
        from datasets.emnist import get_loader_emnist as get_loader
    if args.dataset in ['fashion']:
        from datasets.fashion_mnist import get_loader_fashion_mnist as get_loader
    return get_loader(args)


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

def random_patch(imgs, args):
#    patches = list()
    padding = (args.patch_size - args.patch_target)//2
    mask_img = torch.zeros_like(imgs).to(imgs.device)
#    pad = Pad(padding + args.patch_target, fill=-1)
#    imgs = pad(imgs)
#    mask_patch = torch.zeros(1, 1, args.patch_size, args.patch_size).to(imgs.device)
#    mask_patch[:,:, padding: args.patch_size - padding, padding: args.patch_size - padding] = 1
    for i in range(args.patch_num):
        idx = np.random.randint(0, imgs.size(3) + 2 * padding - args.patch_size, (imgs.size(0), 2))
        if not args.patch_rand:
            idx[:,:] = idx[0:1,:] 
        for i, (y1, x1) in enumerate(idx):
            y2 =  y1 + np.random.randint(args.patch_size*2)
            x2 =  x1 + np.random.randint(args.patch_size*2)
#            patches.append(imgs[i:i+1, :, y1:y2, x1:x2])
            mask_img[i, :, max(0,y1-padding):min(y2 - padding,args.img_size), max(0,x1-padding):min(x2-padding,args.img_size)] = 1
    return mask_img
#    return torch.cat(patches,dim=0), mask_patch, mask_img
    
def noisy_labels(y, p_flip):
    # choose labels to flip
    flip_ix = np.random.choice(y.size(0), int(y.size(0) *  p_flip))
    # invert the labels in place
    y[flip_ix] = 1 - y[flip_ix]
    return y


def train(wandb, args, classifier, G, D):
    fixed_z = torch.randn(
            args.test_batch_size, 
            args.latent_size,
            ).to(args.device)
    fixed_z[0,:] = 0.0
    fixed_c = torch.arange(args.n_classes).tile(args.test_batch_size// args.n_classes).to(args.device)
    attack_criterion = LabelSMoothingLoss(
            args.n_classes, smoothing=args.attack_labelsmooth)
    generator_criterion = nn.BCELoss()
    recon_criterion = nn.MSELoss()
    optimizer_g = optim.Adam(G.parameters(), lr =args.lr, betas = (args.beta_1, args.beta_2))
    optimizer_d = optim.Adam(D.parameters(), lr =args.lr, betas = (args.beta_1, args.beta_2))
    train_loader, _, _ = get_data_loader(args= args)
    if args.decay_type == "cosine":
        scheduler_g = WarmupCosineSchedule(optimizer_g, warmup_steps=args.warmup_steps, t_total=args.epochs*len(train_loader))
        scheduler_d = WarmupCosineSchedule(optimizer_d, warmup_steps=args.warmup_steps, t_total=args.epochs*len(train_loader))
    else:
        scheduler_g = WarmupLinearSchedule(optimizer_g, warmup_steps=args.warmup_steps, t_total=args.epochs*len(train_loader))
        scheduler_d = WarmupLinearSchedule(optimizer_d, warmup_steps=args.warmup_steps, t_total=args.epochs*len(train_loader))
    start_epoch = 0
    # We only save the model who uses device "cuda:0"
    # To resume the device for the save model woule be also "cuda:0"
    if args.resume:
        ckpt = load_ckpt(args.ckpt_fpath)
        optimizer_g.load_state_dict(ckpt['optimizer_g'])
        optimizer_d.load_state_dict(ckpt['optimizer_d'])
        scheduler_g.load_state_dict(ckpt['scheduler_g'])
        scheduler_d.load_state_dict(ckpt['scheduler_d'])
        start_epoch = ckpt['epoch']+1
    Loss_g = AverageMeter()
    Loss_d = AverageMeter()
    D_x_total = AverageMeter()
    D_G_z_total = AverageMeter()
    Acc_total = AverageMeter()

    pbar = tqdm(
            range(start_epoch,args.epochs),
            disable = args.local_rank not in [-1,0]
            )
    # Prepare dataset and dataloader
    for epoch in pbar:
        # Switch Training Mode
        G.train()
        D.train()
        tepoch = tqdm(train_loader,
                disable=args.local_rank not in [-1,0])
        for data in tepoch:
            # (1) Update Discriminator (real data)
            optimizer_d.zero_grad()
            inputs = data[0].to(args.device, 
                    non_blocking=True)
#            inputs += LSM = torch.log(1e-6 + sftm(classifier( (fake+1)/2.0 + (0.5 + 0.05*(args.epochs - epoch) / args.epochs) * torch.randn_like(fake, device=fake.device))))
#            patches_org, _, _ = random_patch(inputs, args)
            mask = random_patch(inputs, args)
#            patches_org = inputs * mask_org -1 * (1-mask_org)
#            patches_org = inputs.unfold(2, args.patch_size, args.patch_stride).unfold(3, args.patch_size, args.patch_stride).reshape(-1, args.num_channel, args.patch_size, args.patch_size)
            real_target = torch.ones(mask.size(0)).to(args.device) * 0.9 
            real_target = noisy_labels(real_target, args.p_flip)
            fake_target = torch.zeros(mask.size(0)).to(args.device)
            fake_target = noisy_labels(fake_target, args.p_flip)
#            torch.full((patches_org.size(0),),
#                    real_label,
#                    dtype=torch.float,
#                    device=args.device)
            err_real_d = generator_criterion(D(inputs.detach() * mask).view(-1),real_target - args.gan_labelsmooth)
            if args.local_rank in [-1,0]:
                D_x = D(inputs*mask).view(-1).detach().mean().item()
                D_x_total.update(D_x, mask.size(0))
            err_real_d.backward()
#            optimizer_d.step()

            # (2) update discrminator (fake data)

#            optimizer_d.zero_grad()
            latent_vector = torch.randn(inputs.size(0), args.latent_size).to(args.device)
#            label.fill_(fake_label)
            c = torch.multinomial(torch.ones(args.n_classes), inputs.size(0), replacement=True).to(args.device)
            fake = G(latent_vector, c)

#            patches_fake, mask_fake, mask_fake_img = random_patch(fake, args)
#            patches_fake = fake * mask_fake -1 * (1-mask_fake)
            fake_masked = fake * mask 
#            patches_fake = fake.unfold(2, args.patch_size, args.patch_stride).unfold(3, args.patch_size, args.patch_stride).reshape(-1, args.num_channel, args.patch_size, args.patch_size)
#            outputs = D(patches_fake.detach()).view(-1)
            err_fake_d = generator_criterion(D(fake_masked.detach()).view(-1), fake_target)
            err_fake_d.backward()
            optimizer_d.step()
            if args.local_rank in [-1, 0]:
                err_d = (err_fake_d + err_real_d)/2
                Loss_d.update(err_d.detach().mean().item(), mask.size(0))

            # (3) Generator update
            optimizer_g.zero_grad()
#            label.fill_(real_label)
            outputs = D(fake_masked).view(-1)
            err_g = generator_criterion(outputs, real_target)
            sftm = nn.Softmax(dim = -1)
            target_id = c
#            target_id = args.fixed_id * \
#                    torch.ones(fake.size(0)).to(cfg.device).long()
#            fake_masked = fake * mask_fake_img + fake.detach() * (1-mask_fake_img)
#            LSM = torch.log(1e-6 + sftm(classifier( (fake+1)/2.0 + (0.5 + 0.05*(args.epochs - epoch) / args.epochs) * torch.randn_like(fake, device=fake.device))))
            LSM = torch.log(1e-6 + sftm(classifier((fake+1)/2.0)))
            loss_attack = attack_criterion(LSM, target_id)
            loss_gen = err_g + args.w_attack * loss_attack
            loss_gen.backward()
            optimizer_g.step()

            if args.w_recon > 0.0 :
                fake_new = G(latent_vector, c).detach()
                fake_target = fake_new * mask + fake.detach() * (1-mask)
                for _ in range(3):
                    optimizer_g.zero_grad()
                    fake = G(latent_vector, c)
                    loss_recon =  args.w_recon * recon_criterion(fake, fake_target)
                    loss_recon.backward()
                    optimizer_g.step()


            scheduler_d.step()
            scheduler_g.step()

            train_acc = LSM[torch.arange(c.size(0)),c].exp().mean().item()
            if args.local_rank in [-1, 0]:
                Acc_total.update(train_acc, inputs.size(0))
                Loss_g.update(err_g.detach().item(), inputs.size(0))
                D_G_z_total.update(outputs.mean().item(), inputs.size(0))
                tepoch.set_description(f'Epoch {epoch}: Loss_D : {Loss_d.avg:2.3f}, Loss_G: {Loss_g.avg:2.3f}, lr is {scheduler_d.get_lr()[0]:.2E}, acc:{Acc_total.avg:2.3f}')
        # (4) After end of epoch, save result model
        if epoch % args.eval_every == 0:
            evaluate(wandb, args, classifier, G, fixed_z=fixed_z, fixed_c = fixed_c, epoch = epoch)

        if args.local_rank in [-1,0]:
            model_to_save_g = G.module if hasattr(G, 'module') else G
            model_to_save_d = D.module if hasattr(D, 'module') else D
            ckpt = {
                    'model_g' : model_to_save_g.state_dict(),
                    'model_d' : model_to_save_d.state_dict(),
                    'optimizer_g' : optimizer_g.state_dict(),
                    'optimizer_d' : optimizer_d.state_dict(),
                    'scheduler_g' : scheduler_g.state_dict(),
                    'scheduler_d' : scheduler_d.state_dict(),
                    'epoch': epoch
                    }
            save_ckpt(checkpoint_fpath = args.ckpt_fpath, checkpoint = ckpt, is_best=True)
            if args.wandb_active:
                wandb.log({
                    "loss_D" : Loss_d.avg,
                    "Loss_G" : Loss_g.avg,
                    "D(x)" : D_x_total.avg,
                    "D(G(z))" : D_G_z_total.avg,
                    "val_acc" : Acc_total.avg
                    },
                    step = epoch)
#            print("-"*75+ "\n")
#            print(f'Epoch {epoch}: Loss_D : {Loss_d.avg:2.3f}, Loss_G: {Loss_g.avg:2.3f}, lr is {scheduler_d.get_lr()[0]:.2E}, acc:{Acc_total.avg:2.3f}')
#            print("-"*75+ "\n")
        Loss_g.reset()
        Loss_d.reset()
        D_G_z_total.reset()
        D_x_total.reset()
        if args.multigpu:
            torch.distributed.barrier()


def evaluate(wandb, args, classifier, G, fixed_z=None, fixed_c = None, epoch = 0):
    G.eval()
    if fixed_z == None:
        fixed_z = torch.randn(
                args.test_batch_size, 
                args.latent_size,
                ).to(args.device)
    if fixed_c == None:
        fixed_c = torch.multinomial(torch.ones(args.n_classes), args.test_batch_size, replacement=True).to(args.device)
    val_acc = 0
    if args.local_rank in [-1,0]:
        fake = G(fixed_z, fixed_c)
        pred = classifier( (fake+1)/2.0 )
        fake_y = fixed_c
#        fake_y = args.fixed_id * \
#                torch.ones(cfg.test_batch_size).to(cfg.device)
        val_acc = (pred.max(1)[1] == fake_y).float().mean().item()

        image_array = rearrange(fake, 
                '(b1 b2) c h w -> (b1 h) (b2 w) c', b2 = args.n_classes).cpu().detach().numpy().astype(np.float64)
        images = wandb.Image(image_array, caption=f'Acc is {val_acc*100:2.2f}')
        wandb.log({f'reconstruct':images}, step = epoch)

    return val_acc

def main():
    args = para_config()
#    cfg = edict(yaml.safe_load(open(args.config)))
    if args.multigpu:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        if int(os.environ["RANK"]) !=0:
            args.wandb_active=False
    else:
        args.local_rank = -1

    if args.local_rank in [-1,0] and args.test==False and args.wandb_active:
        wandb.init(project = args.wandb_project,
                   entity = args.wandb_id,
                   config = args,
                   name = f'{args.wandb_name}_lr:{args.lr}',
                   group = f'{args.dataset}_P{args.patch_size}_S{args.patch_target}'
                   )
    if args.local_rank in [-1,0]:
        print(args)
    # set cuda flag
    if not torch.cuda.is_available():
        print("WARNING: You have a CUDA device, so you should probably enable CUDA")
    # Set Automatic Mixed Precision
    # We need to use seeds to make sure that model initialization same
    set_random_seeds(random_seed = args.random_seed)
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    if args.multigpu: 
        torch.distributed.init_process_group(backend="nccl")
        # torch.distributed.init_process_group(backend="gloo")
        args.device = torch.device("cuda:{}".format(args.local_rank))
    else:
        args.device = torch.device("cuda:0")

    # torch.distributed.init_process_group(backend="gloo")

    # Encapsulate the model on the GPU assigned to the current process

    classifier = resnet(
            num_classes= args.n_classes,
            num_channel= args.num_channel,
            ).to(args.device)
    G = Generator(
            img_size = args.img_size,
            latent_size = args.latent_size,
            n_classes = args.n_classes,
            n_gf = args.n_gf,
            levels= args.level_g,
            n_c = args.num_channel,
            ).to(args.device)
    D = Discriminator(
            patch_size = args.img_size,
            n_df = args.n_df,
            levels= args.level_d,
            n_c = args.num_channel
            ).to(args.device)
    G.weight_init()
    D.weight_init()

    if os.path.exists(args.ckpt_fpath_class):
        ckpt = load_ckpt(args.ckpt_fpath_class, is_best = True)
        classifier.load_state_dict(ckpt['model'])
        classifier.eval()
        print(f'{args.ckpt_fpath_class} model is loaded!')
    else:
        raise Exception('there is no generative checkpoint')

    # Load target classifier 
    if args.multigpu:
        G = torch.nn.parallel.DistributedDataParallel(G, device_ids=[args.local_rank], output_device=args.local_rank)
        classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[args.local_rank], output_device=args.local_rank)
        D = torch.nn.parallel.DistributedDataParallel(D, device_ids=[args.local_rank], output_device=args.local_rank)
    # Load common generative model
    if args.resume or args.test:
        ckpt = load_ckpt(args.ckpt_fpath, is_best = args.test)
        G.load_state_dict(ckpt['model_g'])
        D.load_state_dict(ckpt['model_d'])

    if args.test:
        evaluate(wandb, args, classifier, G)
    else:
        train(wandb, args, classifier, G, D)


if __name__ == '__main__':
    main()
