
"""Import default libraries"""
import os
import torch
import argparse
import torch.nn as nn

from utils.base_utils import set_random_seeds, get_accuracy, AverageMeter, WarmupCosineSchedule, WarmupLinearSchedule, load_ckpt, save_ckpt, get_data_loader
import wandb
import os
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from einops import rearrange
from torchvision import transforms
import itertools
from torch.nn import functional as F
from einops.layers.torch import Rearrange, Reduce
from otherMIAs.common_GAN import Generator, Discriminator
from torch.nn.parameter import Parameter




def para_config():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    # hyperparameter setting
    parser.add_argument("--epochs",
            type=int,
            default=200)
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
    parser.add_argument("--num-channel",
            type=int,
            default=1)
    parser.add_argument("--img-size",
            type=int,
            default=32)
    parser.add_argument("--n-classes",
            type=int,
            default=10)
    parser.add_argument("--latent-size",
            type=int,
            default=64)
    parser.add_argument("--n-components",
            type=int,
            default=7)
    parser.add_argument("--lambda-miner-kl",
            type=float,
            default=1e-3)
    parser.add_argument("--lambda-miner-entropy",
            type=float,
            default=0.0)

    parser.add_argument("--n-gf",
            type=int,
            default=64)
    parser.add_argument("--n-df",
            type=int,
            default=64)
    parser.add_argument("--levels",
            type=int,
            default=2)
    # save path configuration
    parser.add_argument("--target-dataset",
            type=str,
            default="mnist",
            help="choose the target dataset")
    parser.add_argument("--aux-dataset",
            type=str,
            default="emnist",
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
    parser.add_argument("--weight-decay",
            type=float,
            default = 5.0e-4)
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
            default=3e-4,
            help = "learning rate")
    

    args = parser.parse_args()
    args.device = torch.device("cuda:0")
    args.ckpt_path = os.path.join(args.experiment_dir, 'classifier', args.target_dataset)
    args.ckpt_path_gan = os.path.join(args.experiment_dir, 'common_gan', args.aux_dataset)
    return args

class ReparameterizedGMM_Linear_Multiclass(nn.Module):
    def __init__(
            self,
            n_z,
            n_components=10,
            n_classes=10):
        super(ReparameterizedGMM_Linear_Multiclass, self).__init__()
        self.n_z = n_z
        self.n_components = n_components
        self.n_classes = n_classes
        self.gmms = [ReparameterizedGMM_Linear(n_z = n_z, n_components = n_components) for _ in range(self.n_classes)]
        for ll, gmm in enumerate(self.gmms):
            for name, p in gmm.named_parameters():
                self.register_parameter(f"gmm_{ll}_{name}", p)
    def forward(self, z, y):
        masks = F.one_hot(y, self.n_classes).float()
        masks = masks.t()
        samps = torch.stack([gmm(z) for gmm in self.gmms])
        x = (masks[..., None] * samps).sum(dim=0)
        return x
    
    def logp(self,x,y):
        n = x.shape[0]
        masks = F.one_hot(y, self.n_classes).float()
        masks = masks.t()
        logps = torch.stack([gmm.logp(x.view(n, -1)) for gmm in self.gmms])
        logp = (masks * logps).sum(dim=0)
        return logp

    def sample(self, y):
        N = y.shape[0]
        return self(torch.randn(N, self.n_z).to(y.device), y)

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
    fixed_c = torch.arange(args.n_classes).unsqueeze(1).tile([1,args.test_batch_size// args.n_classes]).view(-1).to(args.device)
    optimizer = optim.Adam(miner.parameters(), lr =args.lr, betas = (args.beta_1, args.beta_2))
    CE_loss = nn.CrossEntropyLoss()
    if args.decay_type == "cosine":
        scheduler= WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.epochs*100)
    else:
        scheduler= WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.epochs*100)
    # We only save the model who uses device "cuda:0"
    # To resume the device for the save model woule be also "cuda:0"
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
            z = torch.randn(
                    args.train_batch_size,
                    args.latent_size).to(args.device)
            c = torch.randint(
                    args.n_classes,
                    (args.train_batch_size,)).to(args.device)
            w = miner(z, c)
            fake = G(w)
            loss_attack = CE_loss(classifier(fake), c)
            optimizer.zero_grad()
            if args.lambda_miner_entropy > 0:
                loss_attack += args.lambda_miner_entropy * miner.entropy()
            if args.lambda_miner_kl > 0:
                loss_kl = torch.mean(miner.logp(w, c) - gaussian_logp(torch.zeros_like(w), torch.zeros_like(w), w).sum(-1))
                loss_attack += args.lambda_miner_kl * loss_kl
            loss_attack.backward()
            optimizer.step()
            scheduler.step()
            Avg_loss.update(loss_attack.detach().item(), args.train_batch_size)
            train_Acc = (classifier(fake).max(1)[1] == c).float().mean().item()
            Avg_acc.update(train_Acc, args.train_batch_size)
            tepoch.set_description(f'Ep {epoch}: L : {Avg_loss.avg:2.3f}, Acc: {Avg_acc.avg:2.3f}, lr: {scheduler.get_lr()[0]:2.3e}')

        if epoch % args.eval_every == args.eval_every - 1 or epoch==0:
            val_acc, images = evaluate(wandb, args, miner, G, classifier_val, fixed_z, fixed_c, epoch)
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

def evaluate(wandb, args, miner, G, classifier, fixed_z=None, fixed_c = None, epoch = 0):
    if fixed_z == None:
        fixed_z = torch.randn(
                args.test_batch_size,
                args.latent_size,
                ).to(args.device)
    if fixed_c == None:
        fixed_c = torch.randint(
                args.n_classes,
                (args.test_batch_size,)).to(args.device)
    w = miner(fixed_z, fixed_c)
    fake = G(w)
    pred = classifier(fake)
    val_acc = (pred.max(1)[1] == fixed_c).float().mean().item()
    fake = fake.reshape(args.n_classes, args.test_batch_size // args.n_classes, args.num_channel, args.img_size, args.img_size)
    image_array = rearrange(fake,
            'b1 b2 c h w -> (b2 h) (b1 w) c').cpu().detach().numpy().astype(np.float64)
    images = wandb.Image(image_array, caption = f"Epoch {epoch} : {val_acc:2.3f}")
    return val_acc, images



def main():
    args = para_config()
    
    if args.wandb_active:
        wandb.init(project = args.wandb_project,
                   entity = args.wandb_id,
                   config = args,
                   name = f'Dataset: {args.target_dataset}',
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

    # Encapsulate the model on the GPU assigned to the current process
    if args.img_size == 32:
        from models.resnet_32x32 import resnet10 as resnet
        from models.resnet_32x32 import resnet50 as resnet_val
    else:
        from models.resnet import resnet18 as resnet
        from models.resnet import resnet50 as resnet_val
    
    miner = ReparameterizedGMM_Linear_Multiclass(
            n_components = args.n_components, 
            n_classes = args.n_classes,
            n_z = args.latent_size,
            ).to(args.device)
        
    classifier = resnet(
            num_classes= args.n_classes,
            num_channel= args.num_channel,
            ).to(args.device)
    classifier_val = resnet_val(
            num_classes= args.n_classes,
            num_channel= args.num_channel,
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

    train(wandb, args, miner, G, D, classifier, classifier_val)
    wandb.finish()


if __name__ == '__main__':
    main()
