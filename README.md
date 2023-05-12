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
pip install tqdm easydict wandb imageio tqdm
python setup.py develop
conda install -c conda-forge torchmetrics
```

3. Wandb

- WANDB [Enter this link](https://wandb.ai/site)
1. create your account (you can use your Github account)
2. in `config/*****.yaml` edit wandb setting
3. run our script and type certification code.
4. Done

## Implementation

### 1.Train Classifier

- MNIST Classifier
```bash
python classifier/classify_mnist.py # Target Classifier 
python classifier/classify_mnist.py --random-seed=7 --ckpt-fpath="../experiments/classifier/mnist_val" # Val Classifier
```


## 2. Train Attacker

- Auxiliary data : EMNIST / target classifier : MNIST 

```bash
python tools/main.py --epochs=200 --train-batch-size=256 --patch-size=6 --patch-stride=2 --keep-ratio=0.6 --n-gf=64 --n-df=16 --level-g=4 --level-d=3 --w-attack=0.01 --lr=3e-3
```

### License:
This project is licensed under MIT License - see the LICENSE file for details
