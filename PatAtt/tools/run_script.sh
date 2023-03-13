#!/bin/bash

# $1: Configuration file (Classifier)
# $2: Configuration file (Common Generator)
# $3: Configuration file (Attacker)
# $4: Number of target classes

# 1. Train Target Clasifier

python ./classifier/classifiy_mnist.py --conifig $1

# 2. Train Validiation Classifier

python ./classifier/classifiy_mnist.py --conifig $1 --val 1

# 3. Train Common Generator

python ./generator/CGAN.py --config $2 

# 4. Attacker (Iteration)

SET=$(seq 0 10)

for idx in $SET do
	python ./tools/train.py --config $3 --fixed-id $idx
