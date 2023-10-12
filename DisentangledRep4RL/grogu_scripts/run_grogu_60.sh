#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:2
#SBATCH --partition=abhinavlong
#SBATCH --nodelist=grogu-1-24
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/1_24.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/1_24.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export CUDA_VISIBLE_DEVICES=0; python train.py task=metaworld_faucet-open-v2 disentangled_version=V1 latent_dim=4096 seed=10 experiment_id=60 &
export CUDA_VISIBLE_DEVICES=0; python train.py task=metaworld_faucet-open-v2 disentangled_version=V1 latent_dim=4096 seed=11 experiment_id=60 &
export CUDA_VISIBLE_DEVICES=1; python train.py task=metaworld_faucet-open-v2 disentangled_version=V1 latent_dim=4096 seed=12 experiment_id=60 &

wait
