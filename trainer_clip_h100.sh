#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#SBATCH --partition=hopper
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

echo "SLURM_JOBID: " $SLURM_JOBID
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
GPUS_PER_NODE=2
NNODES=$SLURM_NNODES

export JOB_NAME=$SLURM_JOB_NAME
export NCCL_DEBUG=INFO

#module load conda
module --ignore-cache load "conda"
source ~/.bashrc
conda activate hopper
module load ffmpeg
# Print allocated GPUs for debugging
echo "Allocated GPUs:"

srun python distributed_main.py --multiprocessing-distributed --config ./configs/pt/clip_cf_h100.json --experiment egoclip_cf