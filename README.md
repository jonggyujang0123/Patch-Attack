# Patch-Based Variational Model Inversion Attacker

## Setup

1. (optional) Install anaconda

If you don't have anaconda3, you can install by executing the below bash commands
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
bash ~/Desktop/<anacondafile> -b -p 
source anaconda3/bin/activate
```

2. Install Dependencies 
```bash
conda create -n torch python=3.9
conda activate torch
conda install pytorch==1.12.0 torchvision==0.13.0 -c pytorch
pip install tqdm easydict wandb imageio tqdm einops torch-fidelity
python setup.py develop
<!-- conda install -c conda-forge torchmetrics -->
```

## Implementation

### 1. Experiment 1: Attack *EMNIST* dataset using *MNIST* auxiliary dataset

1. Train target/validation classifier 

```bash
python classifier/classifier.py --dataset=emnist --train-batch-size=256 --epochs=30
python classifier/classifier.py --dataset=emnist --train-batch-size=256 --epochs=30 --valid=True
python classifier/classifier.py --dataset=emnist --train-batch-size=256 --test=True
```
2. Train Common Generator

```bash
python otherMIAs/commun_GAN.py --levels=2 --latent-size=64
```

3. Run MIAs

```bash
python otherMIAs/general_MI.py # General MI
python otherMIAs/generative_MI.py # Generative MI
python otherMIAs/VMI.py --lr=2e-4 # Variational MI
python tools/main.py --epochs=200 --train-batch-size=256 --patch-size=6 --patch-stride=2 --keep-ratio=0.6 --n-gf=64 --n-df=16 --level-g=4 --level-d=3 --w-attack=0.01 --lr=3e-3
```

### 2. Experiment 2: Attack *KMNIST* dataset using *MNIST* auxiliary dataset

1. Train target/validation classifier

```bash
python classifier/classifier.py --dataset=kmnist --train-batch-size=256 --epochs=30
python classifier/classifier.py --dataset=kmnist --train-batch-size=256 --epochs=30 --valid=True
python classifier/classifier.py --dataset=kmnist --train-batch-size=256 --test=True
```
2. Train Common Generator (you can skip this since we already train the generator in Experiment 1

```bash
python otherMIAs/commun_GAN.py --levels=2 --latent-size=64
```

3. 


### 3. Experiment 3: Attack *CelebA* dataset using *LFW* dataset
- Download Celeba dataset
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=share_link&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=share_link&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ" -O img_align_celeba.zip && rm -rf /tmp/cookies.txt
```


```bash

```

### License:
This project is licensed under MIT License - see the LICENSE file for details
