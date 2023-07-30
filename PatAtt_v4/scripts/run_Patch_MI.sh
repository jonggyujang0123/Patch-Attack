#!/bin/bash

DATA="$1"
GID="$2"

SET=$(seq 1 10)

for cls in $SET
do
    if [ "$DATA" = "mnist" ]
    then
		command="python tools/main.py --epochs=60 --target-dataset=mnist --aux-dataset=HAN --train-batch-size=128 --lr=3e-4 --n-gf=64 --n-df=256 --level-g=3 --w-attack=8.0 --w-mr=0e-3 --epoch-pretrain=-1 --gan-labelsmooth=0.0 --patch-size=6 --patch-stride=2 --patch-padding=0 --target-class=3 --target-class=$((cls-1))"
    elif [ "$DATA" = "emnist" ]
    then
		command="python tools/main.py --epochs=60 --target-dataset=emnist --aux-dataset=HAN --train-batch-size=128 --lr=3e-4 --n-gf=64 --n-df=256 --level-g=3 --w-attack=8.0 --w-mr=0e-3 --epoch-pretrain=-1 --gan-labelsmooth=0.0 --patch-size=6 --patch-stride=2 --patch-padding=0 --target-class=3 --target-class=$((cls-1))"
    elif [ "$DATA" = "cifar10" ]
    then
		command="--epochs=30 --target-dataset=cifar10 --aux-dataset=cifar100 --train-batch-size=64 --lr=3e-4 --n-gf=64 --n-df=256 --level-g=3 --level-q=3 --w-attack=1.0 --epoch-pretrain=1 --gan-labelsmooth=0.0 --patch-size=6 --patch-stride=2 --patch-padding=0 --target-class=1 --n-disc=2 --w-disc=0.3 --n-cont=3 --w-cont=0.15 --target-class=$((cls-1))"
    fi
    # Execute the command with CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES="$GID" $command
    echo "Done with class $DATA / $cls"
done

