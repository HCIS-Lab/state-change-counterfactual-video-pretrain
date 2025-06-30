#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=700G
#SBATCH --cpus-per-task=20
#SBATCH --time=01:00:00
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
conda activate hiervl
module load ffmpeg
# Print allocated GPUs for debugging
echo "Allocated GPUs:"

srun python distributed_main.py --multiprocessing-distributed --config ./configs/pt/agg_cf_v100.json --experiment egoaggregation