#!/bin/bash

python classifier/classifier.py --dataset=mnist --train-batch-size=256 --epochs=20 --lr=0.025 # target
python classifier/classifier.py --dataset=mnist --train-batch-size=256 --epochs=20 --lr=0.025 --val # validation
python classifier/classifier.py --dataset=mnist --browse # browse dataset


python classifier/classifier.py --dataset=emnist --train-batch-size=256 --epochs=30 --lr=0.025 # target
python classifier/classifier.py --dataset=emnist --train-batch-size=256 --epochs=30 --lr=0.025 --val # validation
python classifier/classifier.py --dataset=emnist --browse # browse datasetz

python classifier/classifier.py --dataset=cifar10 --train-batch-size=256 --epochs=80 --lr=0.025 # Target
python classifier/classifier.py --dataset=cifar10 --train-batch-size=256 --epochs=80 --lr=0.025 --val --cutmix # Validation
python classifier/classifier.py --dataset=cifar10 --browse # browse dataset
