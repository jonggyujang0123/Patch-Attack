# Patch-Based Variational Model Inversion Attacker

My System Configuration:
- i9-12900 cpu
- NVIDIA 3090 GPU
- 128GB RAM

## 1. Setup


### 1.1. (optional) Install anaconda

If you don't have anaconda3, you can install by executing the below bash commands
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
bash ~/Desktop/<anacondafile> -b -p 
source anaconda3/bin/activate
```

### 1.2. Install Dependencies 
```bash
conda create -n torch python=3.9
conda activate torch
conda install pytorch==1.12.0 torchvision==0.13.0 -c pytorch
pip install tqdm easydict wandb imageio tqdm einops torch-fidelity albumentations sentence_transformers einops wandb scipy
pip install pytorch-fid
python setup.py develop
pip install -U git+https://github.com/facebookresearch/fvcore.git
conda install -c conda-forge torchmetrics
pip install -U sentence-transformers
```
### 1.3. Download Dataset
```bash
cd ~/Patch-Attack/data
git clone https://github.com/hslyu/HAN.git
```

### 1.4. Move to the working dir
```bash
cd Patch-Attack/PatAtt_v4
```

## 2. Prepare the Experiments

### 2.1. Train Classifiers

**Run By Script**
```
sh scripts/run_classfiers.sh
```
Results 
|Model|MNIST|EMNIST|CIFAR10|
|---|---|---|---|
|[ResNet-18](https://arxiv.org/abs/1512.03385) (Target)|99.44%|99.58%|95.07%|
|[DLA-34](https://arxiv.org/pdf/1707.06484.pdf) (Validation)|99.58%|95.30%|95.51%|

### 2.2. Train Common Generators
1. Train Gray-scale GAN using HAN dataset.
```bash
python otherMIAs/common_GAN.py --levels=3 --latent-size=128 --dataset=HAN --train-batch-size=128
```
2. Train RGB GAN using CIFAR 100 dataset except the labels related to CIFAR 10 dataset.
```bash
python otherMIAs/common_GAN.py --levels=3 --latent-size=128 --dataset=cifar100 --train-batch-size=128 --epochs=100
```


## 3. Patch-MIA: Experiments

Implemenation Options 
- mnist
- emnist
- cifar10


```bash
sh script/run_Patch_MI.sh {dataset} {gpu_id}
```

## 4. OtherMIAs (GMI, Generative MI, VMI)


### 4.1. GMI
```bash
sh scripts/run_General_MI.sh {dataset} {gpu_id}
```

### 4.2. Generative MI

```bash
sh scripts/run_Generative_MI.sh {dataset} {gpu_id}
```

### 4.3. Variational MI
```bash
sh scripts/run_VMI.sh {dataset} {gpu_id}
```
## 5. Evaluation
att options
- GMI
- GANMI
- VMI
- PMI

By running the above training, result images are saved in `Patch-Attack/PatAtt_v4/Results/{att}/{dataset}`
```bash
sh scripts/run_eval.sh {dataset} {att} {gpu_id}
```

|      |                  | Accuracy (Top1)↑ | Accuracy (Top5)↑ | Confidence ↑ | Precision ↑ | Recall ↑ | Coverage ↑ | Density ↑ | FID ↓    |
|------|------------------|------------------|------------------|--------------|-------------|----------|------------|-----------|----------|
|MNIST | GMI              | 0.4754 | 0.9167 | 0.4574 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 407.7958 |
|      | GANMI            | 0.2775           | 0.8867           | 0.2633       | 0.0041      | 0.1054   | 0.0017     | 0.0008    | 199.6182 |
|      | VMI              | 93.55            | 99.54            | 0.8802       | 0.0001      | 0.0174   | 0.0000     | 0.0001    | 172.5581 |
|      | PMI (ours)       |                  |                  |              | 0.0000      |          |            | 0.0001    |          |
|EMNIST| GMI              | 0.1809 | 0.7436 | 0.1835 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 426.1645 |
|      | GANMI            | 0.1276           | 0.5789           | 0.1400       | 0.0000      | 0.0000   | 0.0000     | 0.0000    | 496.0510 |
|      | VMI              | 0.1534           | 0.5972           | 0.1540       | 0.0028      | 0.0222   | 0.0012     | 0.0009    | 184.9981 |
|      | PMI (ours)       | 0.8887           | 0.9842           | 0.8087       | 0.0015      | 0.0078   | 0.0006     | 0.0005    | 145.6704 |
|CIFAR | GMI              |                  |                  |              |             |          |            |           |          |
|      | GANMI            |                  |                  |              |             |          |            |           |          |
|      | VMI              | 0.1005           | 0.5402           | 0.1077       | 0.0120      | 0.0000   | 0.0040     | 0.0001    | 436.6965 |
|      | PMI (ours)       | 0.0531           | 0.6795           | 0.0966       | 0.3420      | 0.0001   | 0.1677     | 0.0113    | 341.4110 |



<!--
## 99. Notes
### 99.1. Download *CelebA* dataset
- Download Celeba dataset if original download linke is not available. 
```bash
mkdir data/celeba
cd data/celeba
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1I7JByq5cA3jeiEgxwtXAOOi8jwupQvCX' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1I7JByq5cA3jeiEgxwtXAOOi8jwupQvCX" -O celeba.zip && rm -rf ~/cookies.txt
unzip celeba.zip
```
-->

### License:
This project is licensed under MIT License - see the LICENSE file for details
