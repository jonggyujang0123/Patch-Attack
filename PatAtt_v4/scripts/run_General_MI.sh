#!/bin/bash

DATA="$1"
GID="$2"

SET=$(seq 1 10)

for cls in $SET
do
    if [ "$DATA" = "mnist" ]
    then
        command="python otherMIAs/general_MI.py --lr=3e-2 --target-dataset=mnist --target-class=$((cls-1))"
    elif [ "$DATA" = "emnist" ]
    then
        command="python otherMIAs/general_MI.py --lr=3e-2 --target-dataset=emnist --target-class=$((cls-1))"
    elif [ "$DATA" = "cifar10" ]
    then
        command="python otherMIAs/general_MI.py --lr=3e-2 --target-dataset=cifar10 --target-class=$((cls-1))"
    fi
    
    # Execute the command with CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES="$GID" $command
    
    echo "Done with class $DATA / $cls"
done
