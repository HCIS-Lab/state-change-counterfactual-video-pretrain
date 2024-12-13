#!/bin/bash

#SBATCH --mail-user=kung@iu.edu
#SBATCH --mail-type=begin,end,fail
#SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

echo "SLURM_JOBID: " $SLURM_JOBID

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
GPUS_PER_NODE=1
NNODES=$SLURM_NNODES
export JOB_NAME=$SLURM_JOB_NAME

module load conda
source ~/.bashrc
conda activate hiervl
nvidia-smi
srun python distributed_main.py --multiprocessing-distributed --config ./configs/pt/egoclip_cf.json --experiment egoclip_cf
