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
```

3. Wandb

- WANDB [Enter this link](https://wandb.ai/site)
1. create your account (you can use your Github account)
2. in `config/*****.yaml` edit wandb setting
3. run our script and type certification code.
4. Done

## Implementation

### 1.Train Classifier

- Train target classifier
```bash
python classifier/classify_mnist.py
```

- Train validation classifier 
```bash
python classifier/classify_mnist.py --random-seed=7 --ckpt-fpath="/experiments/classifier/mnist_val"
```


## Train/Download Common Generator

## Execute Model Inversion Attacker



## Wandb implementation 

![스크린샷 2022-07-22 오후 5 02 54](https://user-images.githubusercontent.com/88477912/180393054-605830cd-b369-449d-83e9-cfbf73c90aba.png)

** You can turn off the wandb log by editing the below line**
```
os.environ['WANDB_SILENT'] = "true"
```
in agents/mnist.py

# Details:

## Single-GPU 

### Training
```
python tools/main.py --config config/...yaml --resume (0 or 1)
```

### Test

```
python tools/main.py --config config/...yaml --test 1
```

## Multi-GPU 

### Training

```
sh tools/dist_train.sh <ConfigFile> <#GPUS> <1(optional for resume)>  
```

### Test

```
sh tools/dist_test.sh <ConfigFile> <#GPUS>
```

## Results

- CIFAR100 dataset: 77%


### License:
This project is licensed under MIT License - see the LICENSE file for details
