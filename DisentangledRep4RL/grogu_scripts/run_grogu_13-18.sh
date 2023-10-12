#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=124
#SBATCH --time=48:00:00
#SBATCH --mem=700G
#SBATCH --gres=gpu:7
#SBATCH --partition=deepaklong
#SBATCH --nodelist=grogu-1-40
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/1_40.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/1_40.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export CUDA_VISIBLE_DEVICES=0; python train.py task=metaworld_pick-place-v2 disentangled_version=V1 seed=10 experiment_id=18 latent_dim=8912 &
export CUDA_VISIBLE_DEVICES=0; python train.py task=metaworld_pick-place-v2 disentangled_version=V1 seed=11 experiment_id=18 latent_dim=8912 &
export CUDA_VISIBLE_DEVICES=0; python train.py task=metaworld_pick-place-v2 disentangled_version=V1 seed=12 experiment_id=18 latent_dim=8912 &

export CUDA_VISIBLE_DEVICES=1; python train.py task=metaworld_pick-place-v2 disentangled_version=V1 seed=10 experiment_id=13 latent_dim=256 &
export CUDA_VISIBLE_DEVICES=1; python train.py task=metaworld_pick-place-v2 disentangled_version=V1 seed=11 experiment_id=13 latent_dim=256 &
export CUDA_VISIBLE_DEVICES=1; python train.py task=metaworld_pick-place-v2 disentangled_version=V1 seed=12 experiment_id=13 latent_dim=256 &

export CUDA_VISIBLE_DEVICES=2; python train.py task=metaworld_pick-place-v2 disentangled_version=V1 seed=10 experiment_id=14 latent_dim=512 &
export CUDA_VISIBLE_DEVICES=2; python train.py task=metaworld_pick-place-v2 disentangled_version=V1 seed=11 experiment_id=14 latent_dim=512 &
export CUDA_VISIBLE_DEVICES=2; python train.py task=metaworld_pick-place-v2 disentangled_version=V1 seed=12 experiment_id=14 latent_dim=512 &

export CUDA_VISIBLE_DEVICES=3; python train.py task=metaworld_pick-place-v2 disentangled_version=V1 seed=10 experiment_id=15 latent_dim=1024 &
export CUDA_VISIBLE_DEVICES=3; python train.py task=metaworld_pick-place-v2 disentangled_version=V1 seed=11 experiment_id=15 latent_dim=1024 &
export CUDA_VISIBLE_DEVICES=3; python train.py task=metaworld_pick-place-v2 disentangled_version=V1 seed=12 experiment_id=15 latent_dim=1024 &

export CUDA_VISIBLE_DEVICES=4; python train.py task=metaworld_pick-place-v2 disentangled_version=V1 seed=10 experiment_id=16 latent_dim=2048 &
export CUDA_VISIBLE_DEVICES=4; python train.py task=metaworld_pick-place-v2 disentangled_version=V1 seed=11 experiment_id=16 latent_dim=2048 &
export CUDA_VISIBLE_DEVICES=4; python train.py task=metaworld_pick-place-v2 disentangled_version=V1 seed=12 experiment_id=16 latent_dim=2048 &

export CUDA_VISIBLE_DEVICES=5; python train.py task=metaworld_pick-place-v2 disentangled_version=V1 seed=10 experiment_id=17 latent_dim=4096 &
export CUDA_VISIBLE_DEVICES=5; python train.py task=metaworld_pick-place-v2 disentangled_version=V1 seed=11 experiment_id=17 latent_dim=4096 &
export CUDA_VISIBLE_DEVICES=5; python train.py task=metaworld_pick-place-v2 disentangled_version=V1 seed=12 experiment_id=17 latent_dim=4096 &

wait $!
