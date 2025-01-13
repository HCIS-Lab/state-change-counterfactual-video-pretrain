#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
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

# Print allocated GPUs for debugging
echo "Allocated GPUs:"
srun --nodes=2 --ntasks=2 bash -c 'hostname; nvidia-smi'

# srun python distributed_main.py --multiprocessing-distributed --config ./configs/pt/egoclip_cf.json --experiment egoclip_cf

# srun python -m torch.distributed.launch \
#      --nproc_per_node=4 \
#      --nnodes=2 \
#      --node_rank=$SLURM_PROCID \
#      --master_addr=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) \
#      --master_port=$(shuf -i 20000-65000 -n 1) \
#      distributed_main.py --multiprocessing-distributed --config ./configs/pt/egoclip_cf.json --experiment egoclip_cf