# Import Required Libraries
import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from models.resnet_32x32 import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from models.densenet import DenseNet121, DenseNet161, DenseNet169, DenseNet201
from models.dla import DLA
from utils.base_utils import get_data_loader, AverageMeter, accuracy
import numpy as np
import pytorch_fid.fid_score
import torch
from pytorch_fid.inception import InceptionV3
import wandb

def argparser():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--off-aug",
            action=argparse.BooleanOptionalAction,
            default=False)
    parser.add_argument("--off-patch",
            action=argparse.BooleanOptionalAction,
            default=False)
    parser.add_argument('--attacker',
                        type=str,
                        help='GMI | GANMI | VMI | PMI')
    parser.add_argument('--target-dataset', 
                        type=str, 
                        help='mnist | cifar10 | lsun | imagenet | folder | lfw | fake')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='number of workers')
    parser.add_argument("--wandb-project",
            type=str,
            default="PMIA-Evaluate")
    parser.add_argument("--wandb-id",
            type=str,
            default="jonggyujang0123")
    parser.add_argument("--wandb-active",
            type=bool,
            default=True)

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    if args.attacker == "GMI":
        args.attacker_name = "General_MI"
    elif args.attacker == "GANMI":
        args.attacker_name = "Generative_MI"
    elif args.attacker == "VMI":
        args.attacker_name = "Variational_MI"
    elif args.attacker == "PMI":
        args.attacker_name = 'Patch_MI'
    else:
        raise NotImplementedError
    if args.off_patch:
        args.attacker_name = args.attacker_name + '_32'
    if args.off_aug:
        args.attacker_name = args.attacker_name + '_noaug'
    if args.target_dataset == "mnist":
        args.num_channel = 1
        args.img_size = 32
        args.num_classes = 10
    elif args.target_dataset == "emnist":
        args.num_channel = 1
        args.img_size = 32
        args.num_classes = 26
    elif args.target_dataset == "cifar10":
        args.num_channel = 3
        args.img_size = 32
        args.num_classes = 10
    else:
        raise NotImplementedError

    return args

def acc_metric(args, dataset, model):
    model.eval()
    data_loader = DataLoader(dataset, 
                             batch_size=args.batch_size, 
                             shuffle=False, 
                             num_workers=args.num_workers)
    ACC_t1 = AverageMeter()
    Conf = AverageMeter()
    ACC_t5 = AverageMeter()
    for step, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        outputs = model(inputs)
        top1, top5 = accuracy(outputs, targets, topk=(1, 5))
        confidence = outputs.softmax(1)[range(outputs.size(0)), targets].mean()
        ACC_t1.update(top1.item(), inputs.size(0))
        ACC_t5.update(top5.item(), inputs.size(0))
        Conf.update(confidence.item(), inputs.size(0))
    return ACC_t1.avg, ACC_t5.avg, Conf.avg

import numpy as np
import pytorch_fid.fid_score
import torch
from pytorch_fid.inception import InceptionV3

def SingleClassSubset(dataset, cls):
    indices = np.where(np.array(dataset.targets) == cls)[0]
    return Subset(dataset, indices)

class PRCD:
    def __init__(self, args, dataset_real, dataset_fake):
        self.dataset_real = dataset_real
        self.dataset_fake = dataset_fake
        self.batch_size = args.batch_size
        self.dims = 2048
        self.num_workers = args.num_workers
        self.device = args.device
        self.num_classes = args.num_classes
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
        inception_model = InceptionV3([block_idx])
        self.inception_model = inception_model.to(self.device)
        self.up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=True).to(self.device)
    def compute_metric(self, k=3):
        precision_list = []
        recall_list = []
        density_list = []
        coverage_list = []
        for step, cls in tqdm(enumerate(range(10))):
            with torch.no_grad():
                embedding_fake = self.compute_embedding(self.dataset_fake, cls)
                embedding_real = self.compute_embedding(self.dataset_real, cls)
                print(embedding_fake.shape, embedding_real.shape)
                if embedding_fake.shape[0] > embedding_real.shape[0]:
                    embedding_fake = embedding_fake.repeat(embedding_fake.shape[0] // embedding_real.shape[0], 1)[:embedding_real.shape[0]]
                else:
                    embedding_real = embedding_real.repeat(embedding_real.shape[0] // embedding_fake.shape[0], 1)[:embedding_fake.shape[0]]
                #  embedding_fake = embedding_fake[:length]
                #  embedding_real = embedding_real[:length]
                
                pair_dist_real = torch.cdist(embedding_real, embedding_real, p=2)
                pair_dist_real = torch.sort(pair_dist_real, dim=1, descending=False)[0]
                pair_dist_fake = torch.cdist(embedding_fake, embedding_fake, p=2)
                pair_dist_fake = torch.sort(pair_dist_fake, dim=1, descending=False)[0]
                radius_real = pair_dist_real[:, k]
                radius_fake = pair_dist_fake[:, k]

                # Compute precision
                distances_fake_to_real = torch.cdist(embedding_fake, embedding_real, p=2)
                min_dist_fake_to_real, nn_real = distances_fake_to_real.min(dim=1)
                precision = (min_dist_fake_to_real <= radius_real[nn_real]).float().mean()
                precision_list.append(precision.cpu().item())

                # Compute recall
                distances_real_to_fake = torch.cdist(embedding_real, embedding_fake, p=2)
                min_dist_real_to_fake, nn_fake = distances_real_to_fake.min(dim=1)
                recall = (min_dist_real_to_fake <= radius_fake[nn_fake]).float().mean()
                recall_list.append(recall.cpu().item())

                # Compute density
                num_samples = distances_fake_to_real.shape[0]
                sphere_counter = (distances_fake_to_real <= radius_real.repeat(num_samples, 1)).float().sum(dim=0).mean()
                density = sphere_counter / k
                density_list.append(density.cpu().item())

                # Compute coverage
                num_neighbors = (distances_fake_to_real <= radius_real.repeat(num_samples, 1)).float().sum(dim=0)
                coverage = (num_neighbors > 0).float().mean()
                coverage_list.append(coverage.cpu().item())

        # Compute mean over targets
        precision = np.mean(precision_list)
        recall = np.mean(recall_list)
        density = np.mean(density_list)
        coverage = np.mean(coverage_list)
        return precision, recall, density, coverage

    def compute_embedding(self, dataset, cls=None):
        self.inception_model.eval()
        if cls is not None:
            dataset = SingleClassSubset(dataset, cls)
        else:
            raise NotImplementedError
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=False,
                                                 drop_last=False,
                                                 pin_memory=True,
                                                 num_workers=self.num_workers)
        pred_arr = np.empty((len(dataset), self.dims))
        start_idx = 0
        max_iter = int(len(dataset) / self.batch_size)
        for step, (x, y) in enumerate(dataloader):
            x = x.to(self.device)
            with torch.no_grad():
                if x.shape[1] == 1:
                    x = x.repeat(1, 3, 1, 1)
                x = self.up(x)
                pred = self.inception_model(x)[0]
            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            pred_arr[start_idx:start_idx + pred.shape[0]] = pred
            start_idx = start_idx + pred.shape[0]

        return torch.from_numpy(pred_arr)
    
    
class FID_Score:
    def __init__(self, args, dataset_1, dataset_2):
        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
        self.batch_size = args.batch_size
        self.dims = 2048
        self.num_workers = args.num_workers
        self.device = args.device
        self.num_classes = args.num_classes
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
        inception_model = InceptionV3([block_idx])
        self.inception_model = inception_model.to(self.device)
        self.up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=True).to(self.device)
            
    def compute_fid(self):
        m1, s1 = self.compute_statistics(self.dataset_1)
        m2, s2 = self.compute_statistics(self.dataset_2)
        fid_value = pytorch_fid.fid_score.calculate_frechet_distance(
            m1, s1, m2, s2)
        return fid_value

    def compute_statistics(self, dataset):
        self.inception_model.eval()
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=False,
                                                 drop_last=False,
                                                 pin_memory=True,
                                                 num_workers=self.num_workers)
        pred_arr = np.empty((len(dataset), self.dims))
        start_idx = 0
        max_iter = int(len(dataset) / self.batch_size)
        for step, (x, y) in tqdm(enumerate(dataloader)):
            with torch.no_grad():
                x = x.to(self.device)
                if x.shape[1] == 1:
                    x = x.repeat(1, 3, 1, 1)
                x = self.up(x)
                pred = self.inception_model(x)[0]
            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            pred_arr[start_idx:start_idx + pred.shape[0]] = pred
            start_idx = start_idx + pred.shape[0]

        mu = np.mean(pred_arr, axis=0)
        sigma = np.cov(pred_arr, rowvar=False)
        return mu, sigma
    
    
def main(args):
    if args.wandb_active:
        wandb.init(project = args.wandb_project,
                   entity = args.wandb_id,
                   config = args,
                   name = f'{args.target_dataset}_{args.attacker_name}',
                   group = f'{args.target_dataset}'
                   )
    else:
        os.environ['WANDB_MODE'] = 'dryrun'
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.Grayscale(1) if args.num_channel == 1 else transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) if args.num_channel == 3 else transforms.Normalize((0.5,),(0.5,))
        ])
    os.chdir('/home/jgjang/Patch-Attack/PatAtt_v4/')
    if args.target_dataset == 'mnist':
        target_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    elif args.target_dataset == 'emnist':   
        target_dataset = datasets.EMNIST('../data', split='letters',train=True, download=True, transform=transform)
        for i in range(len(target_dataset)):
            target_dataset.targets[i] = target_dataset.targets[i] - 1
    elif args.target_dataset == 'cifar10':
        target_dataset = datasets.CIFAR10('../data', train=True, download=True, transform=transform)
    else:
        raise NotImplementedError
    target_classifier = DLA(num_classes = args.num_classes, num_channel = args.num_channel).to(args.device)
    target_classifier.load_state_dict(torch.load(f'../experiments/classifier/{args.target_dataset}_valid/best.pt')['model'])
    attack_dataset = datasets.ImageFolder(f'./Results/{args.attacker_name}/{args.target_dataset}', transform=transform)
    # compute variance

    Acc_t1, Acc_t5, Confidence = acc_metric(args, attack_dataset, target_classifier)
    print(f'Accuracy Top1: {Acc_t1:.4f} | Accuracy Top5: {Acc_t5:.4f} | Confidence: {Confidence:.4f}')
    prcd = PRCD(args, target_dataset, attack_dataset)
    Precision, Recall, Coverage, Density = prcd.compute_metric()
    print(f'Precision: {Precision:.4f} | Recall: {Recall:.4f} | Coverage: {Coverage:.4f} | Density: {Density:.4f}')
    fid = FID_Score(args,target_dataset, attack_dataset)
    fid_score = fid.compute_fid()
    print(f'FID: {fid_score:.4f}')
    if args.wandb_active:
        wandb.log({
            'Accuracy Top1': Acc_t1,
            'Accuracy Top5': Acc_t5,
            'Confidence': Confidence,
            'Precision': Precision,
            'Recall': Recall,
            'Coverage': Coverage,
            'Density': Density,
            'FID': fid_score
            })
    print(f'| {Acc_t1:.4f} | {Acc_t5:.4f} | {Confidence:.4f} | {Precision:.4f} | {Recall:.4f} | {Coverage:.4f} | {Density:.4f} | {fid_score:.4f} |')

if __name__ == '__main__':
    args = argparser()
    print(f'dataset: {args.target_dataset} | attacker: {args.attacker_name}')
    main(args)

