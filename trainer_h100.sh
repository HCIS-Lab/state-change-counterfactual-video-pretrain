#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#SBATCH --partition=hopper
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=400G
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

echo "SLURM_JOBID: " $SLURM_JOBID
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
GPUS_PER_NODE=4
NNODES=$SLURM_NNODES

export JOB_NAME=$SLURM_JOB_NAME
export NCCL_DEBUG=INFO
# export NCCL_IB_IFNAME=mlx5_0   # or the interface name in 'ibstat'
# export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
module load ffmpeg/4.0.2

#module load conda
module --ignore-cache load "conda"
source ~/.bashrc
conda activate hopper
module load ffmpeg
# Print allocated GPUs for debugging
echo "Allocated GPUs:"

srun python distributed_main.py --multiprocessing-distributed --config ./configs/pt/agg_cf_h100.json --experiment egoaggregation