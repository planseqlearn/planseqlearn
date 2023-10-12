#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --time=48:00:00
#SBATCH --mem=448G
#SBATCH --gres=gpu:8
#SBATCH --partition=deepaklong
#SBATCH --nodelist=grogu-0-19
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/0_19.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/0_19.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export CUDA_VISIBLE_DEVICES=0; python train.py task=metaworld_handle-press-side-v2 disentangled_version=V1 latent_dim=4096 seed=10 experiment_id=66 &
export CUDA_VISIBLE_DEVICES=0; python train.py task=metaworld_handle-press-side-v2 disentangled_version=V1 latent_dim=4096 seed=11 experiment_id=66 &
export CUDA_VISIBLE_DEVICES=1; python train.py task=metaworld_handle-press-side-v2 disentangled_version=V1 latent_dim=4096 seed=12 experiment_id=66 &
export CUDA_VISIBLE_DEVICES=1; python train.py task=metaworld_handle-press-v2 disentangled_version=original_drqv2 seed=10 experiment_id=67 &
export CUDA_VISIBLE_DEVICES=2; python train.py task=metaworld_handle-press-v2 disentangled_version=original_drqv2 seed=11 experiment_id=67 &
export CUDA_VISIBLE_DEVICES=2; python train.py task=metaworld_handle-press-v2 disentangled_version=original_drqv2 seed=12 experiment_id=67 &
export CUDA_VISIBLE_DEVICES=3; python train.py task=metaworld_handle-press-v2 disentangled_version=V1 latent_dim=4096 seed=10 experiment_id=68 &
export CUDA_VISIBLE_DEVICES=3; python train.py task=metaworld_handle-press-v2 disentangled_version=V1 latent_dim=4096 seed=11 experiment_id=68 &
export CUDA_VISIBLE_DEVICES=4; python train.py task=metaworld_handle-press-v2 disentangled_version=V1 latent_dim=4096 seed=12 experiment_id=68 &
export CUDA_VISIBLE_DEVICES=4; python train.py task=metaworld_handle-pull-side-v2 disentangled_version=original_drqv2 seed=10 experiment_id=69 &
export CUDA_VISIBLE_DEVICES=5; python train.py task=metaworld_handle-pull-side-v2 disentangled_version=original_drqv2 seed=11 experiment_id=69 &
export CUDA_VISIBLE_DEVICES=5; python train.py task=metaworld_handle-pull-side-v2 disentangled_version=original_drqv2 seed=12 experiment_id=69 &
export CUDA_VISIBLE_DEVICES=6; python train.py task=metaworld_handle-pull-side-v2 disentangled_version=V1 latent_dim=4096 seed=10 experiment_id=70 &
export CUDA_VISIBLE_DEVICES=6; python train.py task=metaworld_handle-pull-side-v2 disentangled_version=V1 latent_dim=4096 seed=11 experiment_id=70 &
export CUDA_VISIBLE_DEVICES=7; python train.py task=metaworld_handle-pull-side-v2 disentangled_version=V1 latent_dim=4096 seed=12 experiment_id=70 &

wait
