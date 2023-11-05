#!/bin/bash

python classifier/classifier.py --dataset=mnist --train-batch-size=64 --epochs=20 --lr=0.025 # target
python classifier/classifier.py --dataset=mnist --train-batch-size=64 --epochs=20 --lr=0.025 --val # validation
python classifier/classifier.py --dataset=mnist --browse # browse dataset 


python classifier/classifier.py --dataset=emnist --train-batch-size=64 --epochs=30 --lr=0.025 # target
python classifier/classifier.py --dataset=emnist --train-batch-size=64 --epochs=30 --lr=0.025 --val # validation
python classifier/classifier.py --dataset=emnist --browse # browse datasetz

python classifier/classifier.py --dataset=cifar10 --train-batch-size=64 --epochs=80 --lr=0.025 --cutmix # Target
python classifier/classifier.py --dataset=cifar10 --train-batch-size=64 --epochs=80 --lr=0.025 --cutmix --val # Validation
python classifier/classifier.py --dataset=cifar10 --browse # browse dataset

python classifier/classifier.py --dataset=celeba --train-batch-size=16 --epochs=50 --lr=0.01
python classifier/classifier.py --dataset=celeba --train-batch-size=16 --epochs=50 --lr=0.01 --val
python classifier/classifier.py --dataset=celeba --browse
