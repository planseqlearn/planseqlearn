#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=36
#SBATCH --time=48:00:00
#SBATCH --mem=200G
#SBATCH --gres=gpu:3
#SBATCH --partition=deepaklong
#SBATCH --nodelist=grogu-0-24
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/0_24.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/0_24.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export CUDA_VISIBLE_DEVICES=0; python train.py task=metaworld_handle-pull-v2 disentangled_version=original_drqv2 seed=10 experiment_id=71 &
export CUDA_VISIBLE_DEVICES=0; python train.py task=metaworld_handle-pull-v2 disentangled_version=original_drqv2 seed=11 experiment_id=71 &
export CUDA_VISIBLE_DEVICES=1; python train.py task=metaworld_handle-pull-v2 disentangled_version=original_drqv2 seed=12 experiment_id=71 &
export CUDA_VISIBLE_DEVICES=1; python train.py task=metaworld_handle-pull-v2 disentangled_version=V1 latent_dim=4096 seed=10 experiment_id=72 &
export CUDA_VISIBLE_DEVICES=2; python train.py task=metaworld_handle-pull-v2 disentangled_version=V1 latent_dim=4096 seed=11 experiment_id=72 &
export CUDA_VISIBLE_DEVICES=2; python train.py task=metaworld_handle-pull-v2 disentangled_version=V1 latent_dim=4096 seed=12 experiment_id=72 &
wait
