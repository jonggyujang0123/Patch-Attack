import sys

import numpy as np
import torch
from pytorch_fid.inception import InceptionV3

sys.path.insert(0, '/workspace')
from datasets.custom_subset import SingleClassSubset
from utils.stylegan import create_image


class PRCD:
    def __init__(self, dataset_real, dataset_fake, device, crop_size=None, generator=None, batch_size=128, dims=2048, num_workers=16, gpu_devices=[]):
        self.dataset_real = dataset_real
        self.dataset_fake = dataset_fake
        self.batch_size = batch_size
        self.dims = dims
        self.num_workers = num_workers
        self.device = device
        self.generator = generator
        self.crop_size = crop_size

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
        inception_model = InceptionV3([block_idx])
        if len(gpu_devices) > 1:
            self.inception_model = torch.nn.DataParallel(inception_model, device_ids=gpu_devices)
        else:
            self.inception_model = inception_model
        self.inception_model.to(self.device)
        
    def compute_metric(self, num_classes, k=3, rtpt=None):
        precision_list = []
        recall_list = []
        density_list = []
        coverage_list = []
        for step, cls in enumerate(range(num_classes)):
            with torch.no_grad():
                embedding_fake = self.compute_embedding(self.dataset_fake, cls)
                embedding_real = self.compute_embedding(self.dataset_real, cls)
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
                # Update rtpt
                if rtpt:
                    rtpt.step(
                        subtitle=f'PRCD Computation step {step} of {num_classes}')

        # Compute mean over targets
        precision = np.mean(precision_list)
        recall = np.mean(recall_list)
        density = np.mean(density_list)
        coverage = np.mean(coverage_list)
        return precision, recall, density, coverage

    def compute_embedding(self, dataset, cls=None):
        self.inception_model.eval()
        if cls:
            dataset = SingleClassSubset(dataset, cls)
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
            with torch.no_grad():
                if x.shape[1] != 3:
                    x = create_image(x, self.generator,
                                     crop_size=self.crop_size, resize=299, batch_size=int(self.batch_size / 2))

                x = x.to(self.device)
                pred = self.inception_model(x)[0]
            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            pred_arr[start_idx:start_idx + pred.shape[0]] = pred
            start_idx = start_idx + pred.shape[0]

        return torch.from_numpy(pred_arr)
import numpy as np
import pytorch_fid.fid_score
import torch
from pytorch_fid.inception import InceptionV3
from utils.stylegan import create_image

IMAGE_EXTENSIONS = ('bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp')


class FID_Score:
    def __init__(self, dataset_1, dataset_2, device, crop_size=None, generator=None, batch_size=128, dims=2048, num_workers=8, gpu_devices=[]):
        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
        self.batch_size = batch_size
        self.dims = dims
        self.num_workers = num_workers
        self.device = device
        self.generator = generator
        self.crop_size = crop_size

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
        inception_model = InceptionV3([block_idx]).to(self.device)
        if len(gpu_devices) > 1:
            self.inception_model = torch.nn.DataParallel(inception_model, device_ids=gpu_devices)
        else:
            self.inception_model = inception_model
        self.inception_model.to(device)
            
    def compute_fid(self, rtpt=None):
        m1, s1 = self.compute_statistics(self.dataset_1, rtpt)
        m2, s2 = self.compute_statistics(self.dataset_2, rtpt)
        fid_value = pytorch_fid.fid_score.calculate_frechet_distance(
            m1, s1, m2, s2)
        return fid_value

    def compute_statistics(self, dataset, rtpt=None):
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
        for step, (x, y) in enumerate(dataloader):
            with torch.no_grad():
                if x.shape[1] != 3:
                    x = create_image(x, self.generator,
                                     crop_size=self.crop_size, resize=299, batch_size=int(self.batch_size / 2))

                x = x.to(self.device)
                pred = self.inception_model(x)[0]
            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            pred_arr[start_idx:start_idx + pred.shape[0]] = pred
            start_idx = start_idx + pred.shape[0]

            if rtpt:
                rtpt.step(
                    subtitle=f'FID Score Computation step {step} of {max_iter}')

        mu = np.mean(pred_arr, axis=0)
        sigma = np.cov(pred_arr, rowvar=False)
        return mu, sigma
import torch
import torchvision.transforms as T
from datasets.celeba import CelebA1000
from datasets.custom_subset import SingleClassSubset
from datasets.facescrub import FaceScrub
from datasets.stanford_dogs import StanfordDogs
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from torchvision.transforms.transforms import Resize
from utils.stylegan import create_image


class DistanceEvaluation():
    def __init__(self, model, generator, img_size, center_crop_size, dataset, seed):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataset_name = dataset
        self.model = model
        self.center_crop_size = center_crop_size
        self.img_size = img_size
        self.seed = seed
        self.train_set = self.prepare_dataset()
        self.generator = generator

    def prepare_dataset(self):
        # Build the datasets
        if self.dataset_name == 'facescrub':
            transform = T.Compose([
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            train_set = FaceScrub(group='all', train=True,
                                  transform=transform, split_seed=self.seed)
        elif self.dataset_name == 'celeba_identities':
            transform = T.Compose([
                T.Resize(self.img_size),
                T.ToTensor(),
                T.CenterCrop((self.img_size, self.img_size)),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            train_set = CelebA1000(
                train=True, transform=transform, split_seed=self.seed)
        elif 'stanford_dogs' in self.dataset_name:
            transform = T.Compose([
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            train_set = StanfordDogs(
                train=True, cropped=True, transform=transform, split_seed=self.seed)
        else:
            raise RuntimeError(
                f'{self.dataset_name} is no valid dataset name. Chose of of [facescrub, celeba_identities, stanford_dogs].')

        return train_set

    def compute_dist(self, w, targets, batch_size=64, rtpt=None):
        self.model.eval()
        self.model.to(self.device)
        target_values = set(targets.cpu().tolist())
        smallest_distances = []
        mean_distances_list = [['target', 'mean_dist']]
        for step, target in enumerate(target_values):
            mask = torch.where(targets == target, True, False)
            w_masked = w[mask]
            target_subset = SingleClassSubset(self.train_set,
                                              target_class=target)

            target_embeddings = []
            for x, y in DataLoader(target_subset, batch_size):
                with torch.no_grad():
                    x = x.to(self.device)
                    outputs = self.model(x)
                    target_embeddings.append(outputs.cpu())

            attack_embeddings = []
            for w_batch in DataLoader(TensorDataset(w_masked), batch_size, shuffle=False):
                with torch.no_grad():
                    w_batch = w_batch[0].to(self.device)
                    imgs = create_image(w_batch,
                                        self.generator,
                                        crop_size=self.center_crop_size,
                                        resize=(self.img_size, self.img_size),
                                        device=self.device,
                                        batch_size=batch_size)
                    imgs = imgs.to(self.device)
                    outputs = self.model(imgs)
                    attack_embeddings.append(outputs.cpu())

            target_embeddings = torch.cat(target_embeddings, dim=0)
            attack_embeddings = torch.cat(attack_embeddings, dim=0)
            distances = torch.cdist(
                attack_embeddings, target_embeddings, p=2).cpu()
            distances = distances**2
            distances, _ = torch.min(distances, dim=1)
            smallest_distances.append(distances.cpu())
            mean_distances_list.append([target, distances.cpu().mean().item()])

            if rtpt:
                rtpt.step(
                    subtitle=f'Distance Evaluation step {step} of {len(target_values)}')

        smallest_distances = torch.cat(smallest_distances, dim=0)
        return smallest_distances.mean(), mean_distances_list

    def find_closest_training_sample(self, imgs, targets, batch_size=64):
        self.model.eval()
        self.model.to(self.device)
        closest_imgs = []
        smallest_distances = []
        resize = Resize((self.img_size, self.img_size))
        for img, target in zip(imgs, targets):
            img = img.to(self.device)
            img = resize(img)
            if torch.is_tensor(target):
                target = target.cpu().item()
            target_subset = SingleClassSubset(self.train_set,
                                              target_class=target)
            if len(img) == 3:
                img = img.unsqueeze(0)
            target_embeddings = []
            with torch.no_grad():
                # Compute embedding for generated image
                output_img = self.model(img).cpu()
                # Compute embeddings for training samples from same class
                for x, y in DataLoader(target_subset, batch_size):
                    x = x.to(self.device)
                    outputs = self.model(x)
                    target_embeddings.append(outputs.cpu())
            # Compute squared L2 distance
            target_embeddings = torch.cat(target_embeddings, dim=0)
            distances = torch.cdist(output_img, target_embeddings, p=2)
            distances = distances**2
            # Take samples with smallest distances
            distance, idx = torch.min(distances, dim=1)
            smallest_distances.append(distance.item())
            closest_imgs.append(target_subset[idx.item()][0])
        return closest_imgs, smallest_distances

import math

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from utils.stylegan import create_image

from metrics.accuracy import Accuracy, AccuracyTopK

class Accuracy(BaseMetric):
    def __init__(self, name='Accuracy'):
        super().__init__(name)

    def compute_metric(self):
        accuracy = self._num_corrects / self._num_samples
        return accuracy


class AccuracyTopK(BaseMetric):
    def __init__(self, name='accuracy_5', k=5):
        self.k = k
        super().__init__(name)

    def update(self, model_output, y_true):
        y_pred = torch.topk(model_output, dim=1, k=self.k).indices
        num_corrects = 0
        for k in range(self.k):
            num_corrects += torch.sum(y_pred[:, k] == y_true).item()
        self._num_corrects += num_corrects
        self._num_samples += y_true.shape[0]

    def compute_metric(self):
        accuracy = self._num_corrects / self._num_samples
        return accuracy


class ClassificationAccuracy():
    def __init__(self, evaluation_network, device='cuda:0'):
        self.evaluation_network = evaluation_network
        self.device = device

    def compute_acc(self, w, targets, generator, config, batch_size=64, resize=299, rtpt=None):
        self.evaluation_network.eval()
        self.evaluation_network.to(self.device)
        dataset = TensorDataset(w, targets)
        acc_top1 = Accuracy()
        acc_top5 = AccuracyTopK(k=5)
        predictions = []
        correct_confidences = []
        total_confidences = []
        maximum_confidences = []

        max_iter = math.ceil(len(dataset) / batch_size)

        with torch.no_grad():
            for step, (w_batch, target_batch) in enumerate(DataLoader(dataset,
                                                                      batch_size=batch_size,
                                                                      shuffle=False)):
                w_batch, target_batch = w_batch.to(
                    self.device), target_batch.to(self.device)
                imgs = create_image(
                    w_batch, generator, config.attack_center_crop, resize=resize, batch_size=batch_size)
                imgs = imgs.to(self.device)
                output = self.evaluation_network(imgs)

                acc_top1.update(output, target_batch)
                acc_top5.update(output, target_batch)

                pred = torch.argmax(output, dim=1)
                predictions.append(pred)
                confidences = output.softmax(1)
                target_confidences = torch.gather(confidences, 1,
                                                  target_batch.unsqueeze(1))
                correct_confidences.append(
                    target_confidences[pred == target_batch])
                total_confidences.append(target_confidences)
                maximum_confidences.append(torch.max(confidences, dim=1)[0])

            acc_top1 = acc_top1.compute_metric()
            acc_top5 = acc_top5.compute_metric()
            correct_confidences = torch.cat(correct_confidences,
                                            dim=0)
            avg_correct_conf = correct_confidences.mean().cpu().item()
            confidences = torch.cat(total_confidences, dim=0).cpu()
            confidences = torch.flatten(confidences)
            maximum_confidences = torch.cat(maximum_confidences,
                                            dim=0).cpu().tolist()
            avg_total_conf = torch.cat(total_confidences,
                                       dim=0).mean().cpu().item()
            predictions = torch.cat(predictions, dim=0).cpu()

            # Compute class-wise precision
            target_list = targets.cpu().tolist()
            precision_list = [['target', 'mean_conf', 'precision']]
            for t in set(target_list):
                mask = torch.where(targets == t, True, False)
                conf_masked = confidences[mask]
                precision = torch.sum(
                    predictions[mask] == t) / torch.sum(targets == t)
                precision_list.append(
                    [t, conf_masked.mean().item(), precision.cpu().item()])
            confidences = confidences.tolist()
            predictions = predictions.tolist()

            if rtpt:
                rtpt.step(
                    subtitle=f'Classification Evaluation step {step} of {max_iter}')

        return acc_top1, acc_top5, predictions, avg_correct_conf, avg_total_conf, \
            confidences, maximum_confidences, precision_list


def test(wandb, args, classifier_val, G):
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics import StructuralSimilarityIndexMeasure
    #  from sentence_transformers import SentenceTransformer, util
    #  model = SentenceTransformer('clip-ViT-B-32', device = args.device)
    #  import torchvision.transforms as T
    #  from PIL import Image
    #  transform = T.ToPILImage()
    G.eval()
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
    for class_ind in range(args.target_classes):
        dataset = target_dataset[class_ind]
        pbar = tqdm(enumerate(dataset), total=len(dataset))
        fid = FrechetInceptionDistance(feature=64, compute_on_cpu = True).to(args.device)
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0, channel = args.num_channel, compute_on_gpu=True).to(args.device)
        for batch_idx, (x, y) in pbar:
            fixed_c = torch.LongTensor(
                    y.shape[0],
                    ).fill_(class_ind).to(args.device)
            fixed_z = sample_noise(
                    n_disc=args.n_disc,
                    n_classes=args.target_classes,
                    class_ind = fixed_c,
                    n_cont=args.n_cont,
                    n_z = args.latent_size,
                    batch_size=y.shape[0],
                    device=args.device,
                    )
            label = y.to(args.device)
            fake = G(fixed_z)
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
            top1, top5 = accuracy(pred_val, label, topk=(1, 5))
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

def args():
    args = parser.parse_args()
    return args

def main():
    pass

if __name__ == '__main__':
    main()
