"""Import default libraries"""
import os

from utils.base_utils import set_random_seeds, get_accuracy, AverageMeter, WarmupCosineSchedule, WarmupLinearSchedule, load_ckpt, save_ckpt, get_data_loader, noisy_labels
from utils.parameters import para_config
import torch.nn as nn
import torch
import wandb
import os
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from einops import rearrange
from torchvision import transforms
import itertools
"""import target classifier, generator, and discriminator """
from models.resnet_32x32 import resnet10
from models.CGAN import Generator, Discriminator

def train(wandb, args, classifier, classifier_val, G, D):
    fixed_z = torch.randn(
            args.test_batch_size*args.x_sample,
            args.latent_size,
            ).to(args.device)

    fixed_c = torch.arange(args.n_classes).unsqueeze(1).tile([1,args.test_batch_size*args.x_sample// args.n_classes]).view(-1).to(args.device)
    BCE_loss = nn.BCELoss()
    CE_loss = nn.CrossEntropyLoss()
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
    Loss_info = AverageMeter()
    D_x_total = AverageMeter()
    D_G_z_total = AverageMeter()
    Acc_total_t = AverageMeter()
    Max_act = AverageMeter()

    pbar = tqdm(
            range(start_epoch,args.epochs),
            disable = args.local_rank not in [-1,0],
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
                disable=args.local_rank not in [-1,0],
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
            if args.local_rank in [-1,0]:
                D_x_total.update(D_x, d_logit_t.size(0))
            
            # (2) update discrminator (fake data)

            latent_vector = torch.randn(inputs.size(0), args.latent_size).to(args.device)
            c = torch.multinomial(torch.ones(args.n_classes), inputs.size(0), replacement=True).to(args.device)
            fake = G(latent_vector, c)

            d_logit_f = D(fake.detach(), transform = args.transform)
            err_fake_d = BCE_loss(d_logit_f.view(-1), fake_target + args.gan_labelsmooth)
            D_loss = err_real_d + err_fake_d

            D_loss.backward()

            optimizer_d.step()
            if args.local_rank in [-1, 0]:
                err_d = (err_fake_d + err_real_d)
                Loss_d.update(err_d.detach().mean().item(), d_logit_t.size(0))
            # (3) Generator update
            optimizer_g.zero_grad()
            outputs= D.forward(fake, transform = args.transform)
            err_g = BCE_loss(outputs.view(-1) , real_target)

            # (4) MI_loss Loss 
#            optimizer_g.zero_grad()
#            fake = G(latent_vector, c, y_cont_)
            D_disc = classifier(fake)
            loss_attack = CE_loss(D_disc, c)
            info_loss = (w_attack * loss_attack) #/ D.n_patches
            mr_loss = - args.w_mr * ( D_disc.max(1)[0].mean() )
            if epoch > args.epoch_pretrain:
                total_loss = err_g + info_loss + mr_loss #+ mr_loss_avg
            else:
                total_loss = err_g
            total_loss.backward()
            optimizer_g.step()

            scheduler_d.step()
            scheduler_g.step()
            train_acc_target = D_disc.softmax(1)[torch.arange(c.size(0)),c].mean().item() 
            if args.local_rank in [-1, 0]:
                Acc_total_t.update(train_acc_target, inputs.size(0))
                Loss_g.update(err_g.detach().item(), inputs.size(0))
                Loss_info.update(info_loss.detach().item(), inputs.size(0))
                D_G_z_total.update(outputs.mean().item(), inputs.size(0))
                Max_act.update(D_disc.max(1)[0].mean().item(), inputs.size(0))
                tepoch.set_description(f'Ep {epoch}: L_D : {Loss_d.avg:2.3f}, L_G: {Loss_g.avg:2.3f}, L_info: {Loss_info.avg:2.3f}, lr: {scheduler_d.get_lr()[0]:.1E}, Acc:{Acc_total_t.avg:2.3f}, Max_A: {Max_act.avg:2.1f}')
        # (5) After end of epoch, save result model
        if epoch % args.eval_every == args.eval_every - 1 or epoch==0:
            _, images = evaluate(wandb, args, classifier_val, G, fixed_z=fixed_z, fixed_c = fixed_c, epoch = epoch)

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
        if args.multigpu:
            torch.distributed.barrier()


def evaluate(wandb, args, classifier, G, fixed_z=None, fixed_c = None, epoch = 0):
    G.eval()
    if fixed_z == None:
        fixed_z = torch.randn(
                args.test_batch_size*args.x_sample, 
                args.latent_size,
                ).to(args.device)
    if fixed_c == None:
        fixed_c = torch.multinomial(torch.ones(args.n_classes), args.test_batch_size*args.x_sample, replacement=True).to(args.device)
    val_acc = 0
    if args.local_rank in [-1,0]:
        fake = G(fixed_z, fixed_c)
        pred = classifier( fake)
        fake_y = fixed_c
#        fake_y = args.fixed_id * \
#                torch.ones(cfg.test_batch_size).to(cfg.device)
        val_acc = (pred.max(1)[1] == fake_y).float().mean().item()
        pred_sort =  pred[range(fixed_c.shape[0]), fixed_c].reshape(args.n_classes, args.test_batch_size * args.x_sample // args.n_classes).topk(args.test_batch_size // args.n_classes, dim=-1)[1]
        fake = fake.reshape(args.n_classes, args.test_batch_size*args.x_sample//args.n_classes, args.num_channel, args.img_size, args.img_size)[torch.arange(args.n_classes), pred_sort, :, :, :]
        image_array = rearrange(fake, 
                'b1 b2 c h w -> (b2 h) (b1 w) c').cpu().detach().numpy().astype(np.float64)
        images = wandb.Image(image_array, caption=f'Acc is {val_acc*100:2.2f}')
    return val_acc, images

def test(wandb, args, classifier, G, fixed_z=None, fixed_c = None, epoch = 0):
    G.eval()
    if fixed_z == None:
        fixed_z = torch.randn(
                args.test_batch_size*args.x_sample, 
                args.latent_size,
                ).to(args.device)
    if fixed_c == None:
        fixed_c = torch.multinomial(torch.ones(args.n_classes), args.test_batch_size*args.x_sample, replacement=True).to(args.device)


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
                   name = f'P{args.patch_size}, S{args.patch_stride}, w_att:{args.w_attack}, n_df:{args.n_df}, lr:{args.lr}',
                   group = f'{args.dataset}_P{args.patch_size}_S{args.patch_stride}'
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

    classifier = resnet10(
            num_classes= args.n_classes,
            num_channel= args.num_channel,
            ).to(args.device)
    classifier_val = resnet10(
            num_classes= args.n_classes,
            num_channel= args.num_channel,
            ).to(args.device)
    G = Generator(
            img_size = args.img_size,
            latent_size = args.latent_size,
            levels= args.level_g,
            n_classes = args.n_classes,
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
#    G.weight_init()
#    D.weight_init()


    if os.path.exists(args.ckpt_fpath_class):
        ckpt = load_ckpt(args.ckpt_fpath_class, is_best = True)
        classifier.load_state_dict(ckpt['model'])
        classifier.eval()
        print(f'{args.ckpt_fpath_class} model is loaded!')
    else:
        raise Exception('there is no generative checkpoint')

    if os.path.exists(args.ckpt_fpath_class_val):
        ckpt = load_ckpt(args.ckpt_fpath_class_val, is_best = True)
        classifier_val.load_state_dict(ckpt['model'])
        classifier_val.eval()
        print(f'{args.ckpt_fpath_class_val} model is loaded!')
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
#        evaluate(wandb, args,  classifier_val, G)
        test(wandb, args,  classifier_val, G)
    else:
        train(wandb, args, classifier, classifier_val, G, D)
    wandb.finish()


if __name__ == '__main__':
    main()
