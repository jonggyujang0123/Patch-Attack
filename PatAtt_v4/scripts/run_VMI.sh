#!/bin/bash

DATA="$1"
GID="$2"

SET=$(seq 1 10)

for cls in $SET
do
    if [ "$DATA" = "mnist" ]
    then
        command="python otherMIAs/VMI.py --lr=3e-3 --target-dataset=mnist --aux-dataset=HAN --target-class=$((cls-1))"
    elif [ "$DATA" = "emnist" ]
    then
        command="python otherMIAs/VMI.py --lr=3e-3 --target-dataset=emnist --aux-dataset=HAN --target-class=$((cls-1))"
    elif [ "$DATA" = "cifar10" ]
    then
        command="python otherMIAs/VMI.py --lr=1e-2 --target-dataset=cifar10 --aux-dataset=cifar100 --target-class=$((cls-1))"
    elif [ "$DATA" = "celeba" ]
    then
        command="python otherMIAs/VMI.py --lr=1e-2 --target-dataset=celeba --aux-dataset=LFW --target-class=$((cls-1)) --levels=5 --train-batch-size=16"
    fi
    
    # Execute the command with CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES="$GID" $command
    
    echo "Done with class $DATA / $cls"
done
