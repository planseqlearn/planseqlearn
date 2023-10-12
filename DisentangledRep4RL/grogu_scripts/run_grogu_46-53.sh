#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --time=48:00:00
#SBATCH --mem=900G
#SBATCH --gres=gpu:8
#SBATCH --partition=deepaklong
#SBATCH --nodelist=grogu-1-40
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/1_40.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/1_40.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export CUDA_VISIBLE_DEVICES=0; python train.py task=metaworld_door-close-v2 disentangled_version=V1 latent_dim=4096 seed=10 experiment_id=46 &
export CUDA_VISIBLE_DEVICES=0; python train.py task=metaworld_door-close-v2 disentangled_version=V1 latent_dim=4096 seed=11 experiment_id=46 &
export CUDA_VISIBLE_DEVICES=0; python train.py task=metaworld_door-close-v2 disentangled_version=V1 latent_dim=4096 seed=12 experiment_id=46 &

export CUDA_VISIBLE_DEVICES=1; python train.py task=metaworld_door-lock-v2 disentangled_version=original_drqv2 seed=10 experiment_id=47 &
export CUDA_VISIBLE_DEVICES=1; python train.py task=metaworld_door-lock-v2 disentangled_version=original_drqv2 seed=11 experiment_id=47 &
export CUDA_VISIBLE_DEVICES=1; python train.py task=metaworld_door-lock-v2 disentangled_version=original_drqv2 seed=12 experiment_id=47 &

export CUDA_VISIBLE_DEVICES=2; python train.py task=metaworld_door-lock-v2 disentangled_version=V1 latent_dim=4096 seed=10 experiment_id=48 &
export CUDA_VISIBLE_DEVICES=2; python train.py task=metaworld_door-lock-v2 disentangled_version=V1 latent_dim=4096 seed=11 experiment_id=48 &
export CUDA_VISIBLE_DEVICES=2; python train.py task=metaworld_door-lock-v2 disentangled_version=V1 latent_dim=4096 seed=12 experiment_id=48 &

export CUDA_VISIBLE_DEVICES=3; python train.py task=metaworld_door-open-v2 disentangled_version=original_drqv2 seed=10 experiment_id=49 &
export CUDA_VISIBLE_DEVICES=3; python train.py task=metaworld_door-open-v2 disentangled_version=original_drqv2 seed=11 experiment_id=49 &
export CUDA_VISIBLE_DEVICES=3; python train.py task=metaworld_door-open-v2 disentangled_version=original_drqv2 seed=12 experiment_id=49 &

export CUDA_VISIBLE_DEVICES=4; python train.py task=metaworld_door-open-v2 disentangled_version=V1 latent_dim=4096 seed=10 experiment_id=50 &
export CUDA_VISIBLE_DEVICES=4; python train.py task=metaworld_door-open-v2 disentangled_version=V1 latent_dim=4096 seed=11 experiment_id=50 &
export CUDA_VISIBLE_DEVICES=4; python train.py task=metaworld_door-open-v2 disentangled_version=V1 latent_dim=4096 seed=12 experiment_id=50 &

export CUDA_VISIBLE_DEVICES=5; python train.py task=metaworld_door-unlock-v2 disentangled_version=original_drqv2 seed=10 experiment_id=51 &
export CUDA_VISIBLE_DEVICES=5; python train.py task=metaworld_door-unlock-v2 disentangled_version=original_drqv2 seed=11 experiment_id=51 &
export CUDA_VISIBLE_DEVICES=5; python train.py task=metaworld_door-unlock-v2 disentangled_version=original_drqv2 seed=12 experiment_id=51 &

export CUDA_VISIBLE_DEVICES=6; python train.py task=metaworld_door-unlock-v2 disentangled_version=V1 latent_dim=4096 seed=10 experiment_id=52 &
export CUDA_VISIBLE_DEVICES=6; python train.py task=metaworld_door-unlock-v2 disentangled_version=V1 latent_dim=4096 seed=11 experiment_id=52 &
export CUDA_VISIBLE_DEVICES=6; python train.py task=metaworld_door-unlock-v2 disentangled_version=V1 latent_dim=4096 seed=12 experiment_id=52 &

export CUDA_VISIBLE_DEVICES=7; python train.py task=metaworld_hand-insert-v2 disentangled_version=original_drqv2 seed=10 experiment_id=53 &
export CUDA_VISIBLE_DEVICES=7; python train.py task=metaworld_hand-insert-v2 disentangled_version=original_drqv2 seed=11 experiment_id=53 &
export CUDA_VISIBLE_DEVICES=7; python train.py task=metaworld_hand-insert-v2 disentangled_version=original_drqv2 seed=12 experiment_id=53 &

wait
