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
python setup.py develop
pip install -U git+https://github.com/facebookresearch/fvcore.git
conda install -c conda-forge torchmetrics
conda install ipython
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

Or, you can run manually by running the below command.

1. Train MNIST classifiers
    - Target: ResNet18 (accuracy ~ 99.44%)
    - Valid: ResNet34 (accuracy ~ 99.58%)
    - SGD Optimizer with cosineannealing scheduler
```bash
python classifier/classifier.py --dataset=mnist --train-batch-size=64 --epochs=20 --lr=0.25 # target
python classifier/classifier.py --dataset=mnist --train-batch-size=64 --epochs=20 --lr=0.25 --val # validation
python classifier/classifier.py --dataset=mnist --browse # browse dataset 
```

2. Train EMNIST Classifiers
    - Target: ResNet18 (accuracy ~ )
    - Valid: ResNet34
    - Adam Optimizer with linearly decaying learning rate
```bash
python classifier/classifier.py --dataset=emnist --train-batch-size=64 --epochs=30 --lr=0.25 # target
python classifier/classifier.py --dataset=emnist --train-batch-size=64 --epochs=30 --lr=0.25 --val # validation
python classifier/classifier.py --dataset=emnist --browse # browse dataset 
```

3. Train CIFAR10 Classifiers
    - Target: ResNet18
    - Valid: ResNest34
    - Adam Optimizer with linearly decaying learning rate
```bash
python classifier/classifier.py --dataset=cifar10 --train-batch-size=64 --epochs=80 --lr=0.025 --cutmix # Target
python classifier/classifier.py --dataset=cifar10 --train-batch-size=64 --epochs=80 --lr=0.025 --cutmix --val # Validation
python classifier/classifier.py --dataset=cifar10 --browse # browse dataset
```

4. (In revision) Pre-trained facial classification dataset and finetuning
    - Target: ResNext50-32x4d (accuracy: 84.59%)
    - Valid: ResNext50-32x4d (accuracy: 86.76%)
    - SGD optimier with lr 0.01  
    - frequentest 300 classes

```bash
 python classifier/classifier.py --dataset=celeba --train-batch-size=16 --epochs=50 --lr=0.01
 python classifier/classifier.py --dataset=celeba --train-batch-size=16 --epochs=50 --lr=0.01 --val
 python classifier/classifier.py --dataset=celeba --browse

```
### 2.2. Train Common Generators
1. Train Gray-scale GAN using HAN dataset.
```bash
python otherMIAs/common_GAN.py --levels=3 --latent-size=128 --dataset=HAN --train-batch-size=128
```
2. Train RGB GAN using CIFAR 100 dataset except the labels related to CIFAR 10 dataset.
```bash
python otherMIAs/common_GAN.py --levels=3 --latent-size=128 --dataset=cifar100 --train-batch-size=32 --epochs=100
```


## 3. Patch-MIA: Experiments

Implemenation Options 
|Dataset Options|att|
|---|---|
|mnist|GMI|
|emnist|GANMI|
|cifar10|VMI|
| | PMI|

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

By running the above training, result images are saved in `Patch-Attack/PatAtt_v4/Results/{att}/{dataset}`
```bash
sh scripts/run_eval.sh {dataset} {att} {gpu_id}
```



## 99. Notes
### 99.1. Download *CelebA* dataset
- Download Celeba dataset if original download linke is not available. 
```bash
mkdir data/celeba
cd data/celeba
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1I7JByq5cA3jeiEgxwtXAOOi8jwupQvCX' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1I7JByq5cA3jeiEgxwtXAOOi8jwupQvCX" -O celeba.zip && rm -rf ~/cookies.txt
unzip celeba.zip
```


### License:
This project is licensed under MIT License - see the LICENSE file for details
