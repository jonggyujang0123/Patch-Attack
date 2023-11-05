
"""Import default libraries"""
import os
import torch
import argparse
import torch.nn as nn

from utils.base_utils import set_random_seeds, get_accuracy, AverageMeter, WarmupCosineSchedule, WarmupLinearSchedule, load_ckpt, save_ckpt, get_data_loader, accuracy
import wandb
import os
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from einops import rearrange
from torch.nn import functional as F
from einops.layers.torch import Rearrange, Reduce
from otherMIAs.common_GAN import Generator, Discriminator
from torch.nn.parameter import Parameter
from models.resnet_32x32 import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from models.densenet import DenseNet121, DenseNet161, DenseNet169, DenseNet201
from models.dla import DLA


def para_config():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    # hyperparameter setting
    parser.add_argument("--epochs",
            type=int,
            default=31)
    parser.add_argument("--n-images",
            type=int,
            default=1000)
    parser.add_argument("--train-batch-size",
            type=int,
            default=256)
    parser.add_argument("--test-batch-size",
            type=int,
            default=100)
    parser.add_argument("--random-seed",
            type=int,
            default=0)
    parser.add_argument("--eval-every",
            type=int,
            default=5)
    parser.add_argument("--pin-memory",
            type=bool,
            default=True)
    parser.add_argument("--num-workers",
            type=int,
            default=4)
    parser.add_argument("--latent-size",
            type=int,
            default=128)
    parser.add_argument("--n-gf",
            type=int,
            default=64)
    parser.add_argument("--n-df",
            type=int,
            default=64)
    parser.add_argument("--levels",
            type=int,
            default=3)
    parser.add_argument("--n-components",
            type=int,
            default=7)
    parser.add_argument("--lambda-miner-kl",
            type=float,
            default=1e-3)
    parser.add_argument("--lambda-miner-entropy",
            type=float,
            default=0.0)
    parser.add_argument("--kl-every",
            type=int,
            default=5)

    # save path configuration
    parser.add_argument("--target-dataset",
            type=str,
            default="mnist",
            help="choose the target dataset")
    parser.add_argument("--aux-dataset",
            type=str,
            default="HAN",
            help="choose the auxiliary dataset")
    parser.add_argument("--experiment-dir",
            type=str,
            default="../experiments")

    # WANDB SETTINGS
    parser.add_argument("--wandb-project",
            type=str,
            default="Variational-MI")
    parser.add_argument("--wandb-id",
            type=str,
            default="jonggyujang0123")
    parser.add_argument("--wandb-name",
            type=str,
            default="Variational-MI")
    parser.add_argument("--wandb-active",
            type=bool,
            default=True)


    # optimizer setting 
    parser.add_argument("--lr",
            type=float,
            default=2e-4,
            help = "learning rate")
    parser.add_argument("--target-class"
            ,type=int,
            default=0,
            help = "target class")
    args = parser.parse_args()
    args.device = torch.device("cuda:0")
    args.ckpt_path = os.path.join(args.experiment_dir, 'classifier', args.target_dataset)
    args.ckpt_path_gan = os.path.join(args.experiment_dir, 'common_gan', args.aux_dataset)
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
    return args


class ReparameterizedGMM_Linear(nn.Module):
    def __init__(
            self,
            n_z,
            n_components=10):
        super(ReparameterizedGMM_Linear, self).__init__()
        self.n_z = n_z
        self.n_components = n_components
        self.mvns = [ReparameterizedMVN_Linear(n_z = n_z) for _ in range(self.n_components)]
        for ll, mvn in enumerate(self.mvns):
            mvn.m.data = torch.randn_like(mvn.m.data)
            for name, p in mvn.named_parameters():
                self.register_parameter(f"mvn_{ll}_{name}", p)
        self.mixing_weight_logits = Parameter(torch.zeros(self.n_components))

    def sample_components_onehot(self, n):
        return F.gumbel_softmax(self.mixing_weight_logits[None].repeat(n,1), hard=True)

    def forward(self, z):
        bs = z.shape[0]
        masks = self.sample_components_onehot(bs)
        masks = masks.t()
        samps = torch.stack([mvn(z) for mvn in self.mvns])

        x = (masks[..., None] * samps).sum(0)
        return x
        #  return x.view([bs, self.n_z])
    
    def logp(self,x):
        n = x.shape[0]
        logps = []
        for mvn in self.mvns:
            logp = mvn.logp(x.view(n, -1))
            logps.append(logp)
        logps = torch.stack(logps)
        log_mixing_weights = F.log_softmax(self.mixing_weight_logits[None].repeat(n, 1), dim=1).t()
        logp = torch.logsumexp(logps + log_mixing_weights, dim=0) - np.log(self.n_components)
        return logp

    def sample(self, N):
        return self(torch.randn(N, self.n_z).to(self.mvns[0].m.device))

    
class ReparameterizedMVN_Linear(nn.Module):
    def __init__(
            self, 
            n_z
            ):
        super(ReparameterizedMVN_Linear, self).__init__()
        """
        L : std 
        """
        self.n_z = n_z
        self.m = Parameter(torch.randn((1,n_z)))
        self.L = Parameter(
                torch.eye(n_z)
                )

    def forward(self,z):
        """
        args : randomly rampled MVN random variable
        return : mean + std * rv
        """
        return self.m + z @ self.L.T

    def logp(self, x):
        """
        input : x = log of the probability distribution (MVN)
        """
        C = self.L @ self.L.T 
        return torch_mvn_logp_linear(x, self.m, C)

    def entropy(self):
        C = self.L @ self.L.T 
        H = (1/2) * torch.logdet(2*np.pi * np.e * C+1e-3)
        return H

    def sample(self, N):
        return self(torch.randn(N, self.n_z).to(self.m.device))


def torch_mvn_logp_linear(x, m, C):
    """
    Input
        x : (bs, n_z) data
        m : (1, n_z) mean of the data
        C : (n_z, n_z) cov of the data
    output
        (N,) log p = N(x; m, c) 
    """
    k = x.shape[1]
    Z = -(k/2) * np.log(2*np.pi) - (1/2) * torch.logdet(C+1e-3)
    # shape of Z : (pat,)
    Cinv = torch.inverse(C+1e-3)
    ## shape of C : (k, k)
    ## shape of x-m : (bs, pat, k)
    s = -(1/2) *(
            ((x-m) @ Cinv) * (x-m)).sum(-1)

    return Z + s

def gaussian_logp(mean, logstd, x, detach=False):
    """
    lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
    k = 1 (Independent)
    Var = logstd ** 2
    """
    c = np.log(2 * np.pi)
    v = -0.5 * (logstd * 2. + ((x - mean) ** 2) / torch.exp(logstd * 2.) + c)
    if detach:
        v = v.detach()
    return v


def train(wandb, args, miner, G, D, classifier, classifier_val):
    fixed_z = torch.randn(
            args.test_batch_size,
            args.latent_size).to(args.device)
    optimizer = optim.SGD(miner.parameters(), lr =args.lr, momentum = 0.9, weight_decay = 5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * args.epochs, 0.75 * args.epochs], gamma=0.1)
    Avg_loss = AverageMeter()
    Avg_acc = AverageMeter()

    pbar = tqdm(
            range(args.epochs),
            bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
            )
    for epoch in pbar:
        miner.train()
        # Switch Training Mode
        tepoch = tqdm(
                range(100),
                bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
                )
        for data in tepoch:
            optimizer.zero_grad()
            z = torch.randn(
                    args.train_batch_size,
                    args.latent_size).to(args.device)
            w = miner(z)
            fakes = G(w)
            logits = classifier(fakes)
            loss_attack = - logits.softmax(1)[:, args.target_class].log().mean()
            if args.lambda_miner_entropy > 0:
                loss_attack += args.lambda_miner_entropy * miner.entropy()
            if args.lambda_miner_kl > 0 and data%args.kl_every == 0:
                loss_kl = torch.mean(miner.logp(w) - gaussian_logp(torch.zeros_like(w), torch.zeros_like(w), w).sum(-1))
                loss_attack += args.lambda_miner_kl * loss_kl
            loss_attack.backward()
            optimizer.step()
            train_Acc = logits.argmax(1).eq(args.target_class).float().mean().item()
            Avg_loss.update(loss_attack.detach().item(), args.train_batch_size)
            Avg_acc.update(train_Acc, 1)
            tepoch.set_description(f'Ep {epoch}: L : {Avg_loss.avg:2.3f}, Acc: {Avg_acc.avg:2.3f}')
        scheduler.step()
    
        if epoch % args.eval_every == args.eval_every - 1 or epoch==0:
            val_acc, images = evaluate(wandb, args, miner, G, classifier_val, fixed_z, epoch)
            if args.wandb_active:
                wandb.log({
                    "val_acc" : val_acc,
                    "loss" : Avg_loss.avg,
                    "train_acc" : Avg_acc.avg,
                    "image" : images,
                    },
                    step = epoch+1)
        Avg_loss.reset()
        Avg_acc.reset()
    return miner

def evaluate(wandb, args, miner, G, classifier, fixed_z=None, epoch = 0):
    if fixed_z == None:
        fixed_z = torch.randn(
                args.test_batch_size,
                args.latent_size,
                ).to(args.device)
    w = miner(fixed_z)
    img = G(w)
    pred = classifier(img)
    val_acc = (pred.max(1)[1] == args.target_class).float().mean().item()
    if args.num_channel == 1:
        fake = F.pad(img, pad = (1,1,1,1), value = 1)
    else:
        fake = F.pad(img, pad = (1,1,1,1), value = -1)
    image_array = rearrange(fake[:100,...],
                            '(b1 b2) c h w -> (b1 h) (b2 w) c', b1 = 10, b2 = 10).cpu().detach().numpy().astype(np.float64)
    images = wandb.Image(image_array, caption = f"Epoch {epoch} : {val_acc:2.3f}")
    return val_acc, images


def save_images(args, G, miner):
    from torchvision.utils import save_image
    import torchvision
    z = torch.randn(
            args.n_images,
            args.latent_size).to(args.device)
    w = miner(z)
    fake = ( G(w) + 1 ) / 2
    directory = f'./Results/Variational_MI/{args.target_dataset}/{args.target_class}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i in range(args.n_images):
        tensor = fake[i, ...].cpu().detach()
        if args.num_channel == 1:
            tensor = torch.cat([tensor, tensor, tensor], dim = 0)
        save_image(tensor, f'{directory}/{i}.png')

def main():
    args = para_config()
    
    if args.wandb_active:
        wandb.init(project = args.wandb_project,
                   entity = args.wandb_id,
                   config = args,
                   name = f'Dataset: {args.target_dataset} | Class: {args.target_class}',
                   group = f'Dataset: {args.target_dataset}',
                   )
    else:
        os.environ["WANDB_MODE"] = "dryrun"
    print(args)
    # set cuda flag
    if not torch.cuda.is_available():
        print("WARNING: You have a CUDA device, so you should probably enable CUDA")
    # Set Automatic Mixed Precision
    # We need to use seeds to make sure that model initialization same
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
    miner = ReparameterizedGMM_Linear(
            n_components = args.n_components, 
            n_z = args.latent_size,
            ).to(args.device)
    
    # Load Generative model
    G = Generator(
            img_size=args.img_size,
            latent_size = args.latent_size,
            n_gf = args.n_gf,
            levels = args.levels,
            n_c = args.num_channel,
            ).to(args.device)
    D = Discriminator(
            img_size=args.img_size,
            n_df = args.n_df,
            levels = args.levels,
            n_c = args.num_channel
            ).to(args.device)
    if os.path.exists(args.ckpt_path_gan):
        ckpt = load_ckpt(args.ckpt_path_gan, is_best = True)
        G.load_state_dict(ckpt['model_g'])
        G.eval()
        D.load_state_dict(ckpt['model_d'])
        D.eval()
        print(f'{args.ckpt_path_gan} model is loaded!')
    else:
        raise Exception('there is no generative checkpoint')


    miner_trained = train(wandb, args, miner, G, D, classifier, classifier_val)
    save_images(args, G, miner_trained)



if __name__ == '__main__':
    main()
