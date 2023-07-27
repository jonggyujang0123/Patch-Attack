#!/bin/bash

SET=$(seq 0 10)

for idx in $SET do
	echo "====================================="
	echo "Training model $idx"
	echo "====================================="
	python ./tools/train.py 
done

echo "====================================="
echo "Evaluating model"
echo "====================================="
python tools/evaluate.py 


CUDA_VISIBLE_DEVICES=0 python tools/main.py --epochs=60 --target-dataset=emnist --aux-dataset=HAN --train-batch-size=128 --lr=3e-4 --n-gf=64 --n-df=256 --level-g=3 --w-attack=8.0 --w-mr=0e-3 --epoch-pretrain=-1 --gan-labelsmooth=0.0 --patch-size=6 --patch-stride=2 --patch-padding=0 --target-class=3
