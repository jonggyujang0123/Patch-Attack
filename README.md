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
pip install tqdm easydict wandb imageio tqdm einops torch-fidelity albumentations sentence_transformers
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
python otherMIAs/common_GAN.py --levels=2 --latent-size=64 --dataset=mnist --train-batch-size=256
```

3. Run MIAs

```bash
python otherMIAs/general_MI.py --target-dataset=emnist --epochs=50 --lr=3e-3 # General MI
python otherMIAs/generative_MI.py --levels=2 --target-dataset=emnist --aux-dataset=mnist --epochs=400 --lr=3e-2 --random-seed={x} # GenerativeMI
python otherMIAs/VMI.py --lr=2e-4 # Variational MI
python tools/main.py --epochs=200 --target-dataset=emnist --aux-dataset=HAN --train-batch-size=256 --patch-size=8 --patch-stride=4 --patch-padding=4 --n-gf=64 --level-g=2 --latent-size=64 --n-df=192 --lr=3e-3 --w-mr=3e-2 --w-attack=100.0 --latent-size=64 --target-class=1 --gan-labelsmooth=0.0
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
python otherMIAs/common_GAN.py --levels=2 --latent-size=64 --dataset=mnist --train-batch-size=256 # General MI
python otherMIAs/generative_MI.py --levels=2 --target-dataset=kmnist --aux-dataset=mnist --epochs=400 --lr=3e-2 --random-seed={x} # GenerativeMI
```

3. Run MIAs
```bash
python otherMIAs/general_MI.py --target-dataset=kmnist --epochs=50 --lr=3e-3  # General MI
``` 


### 3. Experiment 3: Attack *CelebA* dataset using *LFW* dataset
- Download Celeba dataset if original download linke is not available. (assume you are in this repo not `Pat_att_v3`)
```
mkdir data/celeba
cd data/celeba
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1I7JByq5cA3jeiEgxwtXAOOi8jwupQvCX' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1I7JByq5cA3jeiEgxwtXAOOi8jwupQvCX" -O celeba.zip && rm -rf ~/cookies.txt
unzip celeba.zip
```

1. Train target/validation classifier
```bash
python classifier/classifier.py --dataset=celeba --train-batch-size=256 --epochs=200 --lr=1e-1 
python classifier/classifier.py --dataset=celeba --train-batch-size=256 --epochs=200 --lr=1e-1 --val=True
python classifier/classifier.py --dataset=celeba --train-batch-size=256 --epochs=200 --lr=1e-1 --test=True
```
2. Train Common Generator 
```bash
python otherMIAs/common_GAN.py --levels=4 --latent-size=100 --dataset=LFW --train-batch-size=64 --epochs=200  
```

3. Run MIAs
```bash
python otherMIAs/general_MI.py --target-dataset=celeba --epochs=50 --lr=3e-3  # General MI
python otherMIAs/generative_MI.py --levels=4 --target-dataset=celeba --aux-dataset=LFW --epochs=400 --lr=3e-2 --latent-size=100 --random-seed={xx} # Generative MI
``` 

### License:
This project is licensed under MIT License - see the LICENSE file for details
