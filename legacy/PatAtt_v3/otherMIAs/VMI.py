
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




def para_config():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    # hyperparameter setting
    parser.add_argument("--epochs",
            type=int,
            default=31)
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
    parser.add_argument("--kl-every",
            type=int,
            default=5)

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
    parser.add_argument("--max-classes",
            type=int,
            default=10)
    args = parser.parse_args()
    args.device = torch.device("cuda:0")
    args.ckpt_path = os.path.join(args.experiment_dir, 'classifier', args.target_dataset)
    args.ckpt_path_gan = os.path.join(args.experiment_dir, 'common_gan', args.aux_dataset)
    if args.target_dataset in ['mnist', 'fashionmnist', 'kmnist']:
        args.num_channel = 1
        args.img_size = 32
        args.num_classes = 10
    if args.target_dataset == 'emnist':
        args.num_channel = 1
        args.img_size = 32
        args.num_classes = 26
    if args.target_dataset == 'cifar10':
        args.num_channel = 3
        args.img_size = 32
        args.num_classes = 10
    if args.target_dataset == 'cifar100':
        args.num_channel = 3
        args.img_size = 32
        args.num_classes = 100
    if args.target_dataset == 'LFW':
        args.num_channel = 3
        args.img_size = 64
        args.num_classes = 10
    if args.target_dataset == 'celeba':
        args.num_channel = 3
        args.img_size = 64
        args.num_classes = 1000

    return args

class ReparameterizedGMM_Linear_Multiclass(nn.Module):
    def __init__(
            self,
            n_z,
            n_components=10,
            num_classes=10):
        super(ReparameterizedGMM_Linear_Multiclass, self).__init__()
        self.n_z = n_z
        self.n_components = n_components
        self.num_classes = num_classes
        self.gmms = [ReparameterizedGMM_Linear(n_z = n_z, n_components = n_components) for _ in range(self.num_classes)]
        for ll, gmm in enumerate(self.gmms):
            for name, p in gmm.named_parameters():
                self.register_parameter(f"gmm_{ll}_{name}", p)
    def forward(self, z, y):
        masks = F.one_hot(y, self.num_classes).float()
        masks = masks.t()
        samps = torch.stack([gmm(z) for gmm in self.gmms])
        x = (masks[..., None] * samps).sum(dim=0)
        return x
    
    def logp(self,x,y):
        n = x.shape[0]
        masks = F.one_hot(y, self.num_classes).float()
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
    fixed_c = torch.arange(
            min(args.num_classes, args.max_classes),).unsqueeze(1).tile([1,args.test_batch_size// min(args.num_classes, args.max_classes)]).view(-1).to(args.device)
    optimizer = optim.Adam(miner.parameters(), lr =args.lr, betas = (args.beta_1, args.beta_2))
    CE_loss = nn.CrossEntropyLoss()
    if args.decay_type == "cosine":
        scheduler= WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.epochs*256)
    else:
        scheduler= WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.epochs*256)
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
                range(256),
                bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
                )
        for data in tepoch:
            z = torch.randn(
                    args.train_batch_size,
                    args.latent_size).to(args.device)
            c = torch.randint(
                    min(args.num_classes, args.max_classes),
                    (args.train_batch_size,)).to(args.device)
            w = miner(z, c)
            fake = G(w)
            loss_attack = CE_loss(classifier(fake), c)
            optimizer.zero_grad()
            if args.lambda_miner_entropy > 0:
                loss_attack += args.lambda_miner_entropy * miner.entropy()
            if args.lambda_miner_kl > 0 and data%args.kl_every == 0:
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
    test(wandb, args, miner, G, classifier_val)

def evaluate(wandb, args, miner, G, classifier, fixed_z=None, fixed_c = None, epoch = 0):
    if fixed_z == None:
        fixed_z = torch.randn(
                args.test_batch_size,
                args.latent_size,
                ).to(args.device)
    if fixed_c == None:
        fixed_c = torch.randint(
                min(args.num_classes, args.max_classes),
                (args.test_batch_size,)).to(args.device)
    w = miner(fixed_z, fixed_c)
    fake = G(w)
    pred = classifier(fake)
    val_acc = (pred.max(1)[1] == fixed_c).float().mean().item()
    fake = fake.reshape(min(args.num_classes, args.max_classes), 
                        args.test_batch_size // min(args.num_classes, args.max_classes),
                        args.num_channel, args.img_size, args.img_size)
    image_array = rearrange(fake,
            'b1 b2 c h w -> (b2 h) (b1 w) c').cpu().detach().numpy().astype(np.float64)
    images = wandb.Image(image_array, caption = f"Epoch {epoch} : {val_acc:2.3f}")
    return val_acc, images


def test(wandb, args, miner, G, classifier_val):
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics import StructuralSimilarityIndexMeasure
    #  from sentence_transformers import SentenceTransformer, util
    #  model = SentenceTransformer('clip-ViT-B-32', device = args.device)
    #  import torchvision.transforms as T
    #  from PIL import Image
    #  transform = T.ToPILImage()
    miner.eval()
    Acc_val = AverageMeter() 
    Acc_val_total = []
    Top5_val = AverageMeter()
    Top5_val_total = []
    Conf_val = AverageMeter()
    Conf_val_total = []
    SSIM_val = AverageMeter()
    SSIM_val_total = []
    #  CLIP_val = AverageMeter()
    #  CLIP_val_total = []
    FID_val = AverageMeter()
    FID_val_total = []
    target_dataset, _, _ = get_data_loader(args, args.target_dataset, class_wise = True)
    for class_ind in range(min(args.num_classes, args.max_classes)):
        dataset = target_dataset[class_ind]
        pbar = tqdm(enumerate(dataset), total=len(dataset))
        fid = FrechetInceptionDistance(feature=64, compute_on_cpu = True).to(args.device)
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0, channel = args.num_channel, compute_on_gpu=True).to(args.device)
        for batch_idx, (x, y) in pbar:
            c = torch.tensor([class_ind]).repeat(x.size(0)).to(args.device)
            z = torch.randn(x.size(0), args.latent_size).to(args.device)
            w = miner(z, c)
            fake = G(w)
            #  label = y.to(args.device)
            #  inputs = torch.zeros(label.size(0), min(args.num_classes, args.max_classes)).to(args.device).float()
            fake = fake.detach()
            x = x.to(args.device)
            pred_val = classifier_val(fake)
            pred_val = pred_val.detach()
            x_int  = (255 * (x+1)/2).type(torch.uint8)
            fake_int  = (255 * (fake+1)/2).type(torch.uint8)
            if args.num_channel == 1:
                x_int = x_int.repeat(1,3,1,1)
                fake_int = fake_int.repeat(1,3,1,1)
            fid.update(x_int, real = True)
            fid.update(fake_int, real = False)
            top1, top5 = accuracy(pred_val, c, topk=(1, 5))
            Acc_val.update(top1.item(), x.shape[0])
            Top5_val.update(top5.item(), x.shape[0])
            sftm = pred_val.softmax(dim=1)
            Conf_val.update(sftm[:, class_ind].mean().item(), x.shape[0] )
            #  clip_list = []
            #  for i in range(x.shape[0]):
            #      lists = [transform((x[i,:,:,:]+1)/2), transform((fake[i,:,:,:]+1)/2)]
            #      encoded = model.encode(lists, batch_size=128, convert_to_tensor=True)
            #      score = util.paraphrase_mining_embeddings(encoded)
            #      clip_list.append(score[0][0])
            #  CLIP_val.update(np.mean(clip_list), x.shape[0])
            ssim_list = []
            for i in range(x.shape[0]):
                ssim_score =ssim( (x[i:i+1,:,:,:]+1)/2, (fake[i:i+1,:,:,:]+1)/2)
                ssim_list.append(ssim_score.item())
            SSIM_val.update(np.mean(ssim_list), x.shape[0])
        fid_score = fid.compute()
        fid.reset()
        Acc_val_total.append(Acc_val.avg) 
        Conf_val_total.append(Conf_val.avg)
        SSIM_val_total.append(SSIM_val.avg)
        #  CLIP_val_total.append(CLIP_val.avg)
        FID_val_total.append(fid_score.item())
        Top5_val_total.append(Top5_val.avg)

        print(
            f'==> Testing model.. target class: {class_ind}\n'
            f'    Acc: {Acc_val.avg}\n'
            f'    Top5: {Top5_val.avg}\n'
            f'    Conf: {Conf_val.avg}\n'
            f'    SSIM score : {SSIM_val.avg}\n'
            #  f'    CLIP score : {CLIP_val.avg}\n'
            f'    FID score : {fid_score}\n'
            )
        Acc_val.reset()
        Conf_val.reset()
        #  CLIP_val.reset()
        SSIM_val.reset()
        FID_val.reset()
        Top5_val.reset()
    print(
        f'==> Overall Results\n'
        f'    Acc: {np.mean(Acc_val_total)}\n'
        f'    Conf: {np.mean(Conf_val_total)}\n'
        #  f'    CLIP score : {np.mean(CLIP_val_total)}\n'
        f'    SSIM score : {np.mean(SSIM_val_total)}\n'
        f'    FID score : {np.mean(FID_val_total)}\n'
        )

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
        from models.resnet import resnet34 as resnet
        from models.resnet import resnet50 as resnet_val
    
    miner = ReparameterizedGMM_Linear_Multiclass(
            n_components = args.n_components, 
            num_classes = args.num_classes,
            n_z = args.latent_size,
            ).to(args.device)
        
    classifier = resnet(
            num_classes= args.num_classes,
            num_channel= args.num_channel,
            ).to(args.device)
    classifier_val = resnet_val(
            num_classes= args.num_classes,
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
