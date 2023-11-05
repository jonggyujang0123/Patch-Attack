#!/bin/bash

DATA="$1"
GID="$2"

if [ $3 = "all" ]
then
	SET=$(seq 0 9)
elif [ $3 = "even" ]
then
	SET=$(seq 0 2 8)
elif [ $3 = "odd" ]
then
	SET=$(seq 1 2 9)
else
	SET="$3"
fi



for cls in $SET
do
    if [ "$DATA" = "mnist" ]
    then
		command="python tools/main.py --epochs=30 --target-dataset=mnist --aux-dataset=HAN --train-batch-size=64 --lr=1e-3 --w-attack=30.0 --n-df=256 --patch-size=6 --patch-stride=2 --patch-padding=2 --gan-labelsmooth=0.6 --n-disc=1 --w-disc=0.0 --target-class=$((cls))"
    elif [ "$DATA" = "emnist" ]
    then
		command="python tools/main.py --epochs=30 --target-dataset=emnist --aux-dataset=HAN --train-batch-size=64 --lr=1e-3 --w-attack=30.0 --n-df=256 --patch-size=8 --patch-stride=4 --patch-padding=0 --gan-labelsmooth=0.6 --n-disc=2 --w-disc=0.3 --target-class=$((cls))"
    elif [ "$DATA" = "cifar10" ]
    then
		command="python tools/main.py --epochs=30 --target-dataset=cifar10 --aux-dataset=cifar100 --train-batch-size=64 --lr=1e-3 --w-attack=50.0 --n-df=512 --patch-size=8 --patch-stride=4 --patch-padding=0 --gan-labelsmooth=0.7 --n-disc=1 --w-disc=0.3 --target-class=$((cls))"
    fi
    # Execute the command with CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES="$GID" $command
    echo "Done with class $DATA / $cls"
done

