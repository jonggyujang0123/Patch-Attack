#!/bin/bash
GPUS=$1
RESUME=${2:-0}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}
torchrun --nnodes=$NNODES \
	--node_rank=$NODE_RANK \
	--nproc_per_node=$GPUS \
	--master_addr=$MASTER_ADDR \
	--master_port=$MASTER_PORT \
	tools/main.py \
	--multigpu=True \
	--epoch=100 \
	--train-batch-size=512 \
	--patch-size=5 \
	--patch-target=1\
   	--patch-rand=True \
	--patch-num=4 \
	--w-attack=1.0 \
	--w-recon=0.8 \
	--lr=1e-4 \
	--p-flip=0.15
