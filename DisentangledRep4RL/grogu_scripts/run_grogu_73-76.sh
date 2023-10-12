#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --mem=400G
#SBATCH --gres=gpu:4
#SBATCH --partition=deepaklong
#SBATCH --nodelist=grogu-1-3
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/1_3.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/1_3.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export CUDA_VISIBLE_DEVICES=0; python train.py task=metaworld_lever-pull-v2 disentangled_version=original_drqv2 seed=10 experiment_id=73 &
export CUDA_VISIBLE_DEVICES=0; python train.py task=metaworld_lever-pull-v2 disentangled_version=original_drqv2 seed=11 experiment_id=73 &
export CUDA_VISIBLE_DEVICES=0; python train.py task=metaworld_lever-pull-v2 disentangled_version=original_drqv2 seed=12 experiment_id=73 &

export CUDA_VISIBLE_DEVICES=1; python train.py task=metaworld_lever-pull-v2 disentangled_version=V1 latent_dim=4096 seed=10 experiment_id=74 &
export CUDA_VISIBLE_DEVICES=1; python train.py task=metaworld_lever-pull-v2 disentangled_version=V1 latent_dim=4096 seed=11 experiment_id=74 &
export CUDA_VISIBLE_DEVICES=1; python train.py task=metaworld_lever-pull-v2 disentangled_version=V1 latent_dim=4096 seed=12 experiment_id=74 &

export CUDA_VISIBLE_DEVICES=2; python train.py task=metaworld_peg-insert-side-v2 disentangled_version=original_drqv2 seed=10 experiment_id=75 &
export CUDA_VISIBLE_DEVICES=2; python train.py task=metaworld_peg-insert-side-v2 disentangled_version=original_drqv2 seed=11 experiment_id=75 &
export CUDA_VISIBLE_DEVICES=2; python train.py task=metaworld_peg-insert-side-v2 disentangled_version=original_drqv2 seed=12 experiment_id=75 &

export CUDA_VISIBLE_DEVICES=3; python train.py task=metaworld_peg-insert-side-v2 disentangled_version=V1 latent_dim=4096 seed=10 experiment_id=76 &
export CUDA_VISIBLE_DEVICES=3; python train.py task=metaworld_peg-insert-side-v2 disentangled_version=V1 latent_dim=4096 seed=11 experiment_id=76 &
export CUDA_VISIBLE_DEVICES=3; python train.py task=metaworld_peg-insert-side-v2 disentangled_version=V1 latent_dim=4096 seed=12 experiment_id=76 &
wait
