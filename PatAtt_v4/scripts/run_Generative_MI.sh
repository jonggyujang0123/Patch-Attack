#!/bin/bash

DATA="$1"
GID="$2"

SET=$(seq 1 10)

for cls in $SET
do
    if [ "$DATA" = "mnist" ]
    then
        command="python otherMIAs/generative_MI.py --target-dataset=mnist --aux-dataset=HAN --epochs=40 --lr=1e-2 --target-class=$((cls-1))"
    elif [ "$DATA" = "emnist" ]
    then
        command="python otherMIAs/generative_MI.py --target-dataset=emnist --aux-dataset=HAN --epochs=40 --lr=1e-2 --target-class=$((cls-1))"
    elif [ "$DATA" = "cifar10" ]
    then
        command="python otherMIAs/generative_MI.py --target-dataset=cifar10 --aux-dataset=cifar100 --epochs=40 --lr=3e-2 --target-class=$((cls-1))"
    fi
    
    # Execute the command with CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES="$GID" $command
    
    echo "Done with class $DATA / $cls"
done
