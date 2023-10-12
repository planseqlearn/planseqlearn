#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --time=48:00:00
#SBATCH --mem=900G
#SBATCH --gres=gpu:8
#SBATCH --partition=abhinavlong
#SBATCH --nodelist=grogu-2-3
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/2_3.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/2_3.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export CUDA_VISIBLE_DEVICES=0; python train.py task=metaworld_plate-slide-side-v2 disentangled_version=V1 latent_dim=4096 seed=10 experiment_id=92 &
export CUDA_VISIBLE_DEVICES=0; python train.py task=metaworld_plate-slide-side-v2 disentangled_version=V1 latent_dim=4096 seed=11 experiment_id=92 &
export CUDA_VISIBLE_DEVICES=0; python train.py task=metaworld_plate-slide-side-v2 disentangled_version=V1 latent_dim=4096 seed=12 experiment_id=92 &
export CUDA_VISIBLE_DEVICES=1; python train.py task=metaworld_plate-slide-back-v2 disentangled_version=original_drqv2 seed=10 experiment_id=93 &
export CUDA_VISIBLE_DEVICES=1; python train.py task=metaworld_plate-slide-back-v2 disentangled_version=original_drqv2 seed=11 experiment_id=93 &
export CUDA_VISIBLE_DEVICES=1; python train.py task=metaworld_plate-slide-back-v2 disentangled_version=original_drqv2 seed=12 experiment_id=93 &
export CUDA_VISIBLE_DEVICES=2; python train.py task=metaworld_plate-slide-back-v2 disentangled_version=V1 latent_dim=4096 seed=10 experiment_id=94 &
export CUDA_VISIBLE_DEVICES=2; python train.py task=metaworld_plate-slide-back-v2 disentangled_version=V1 latent_dim=4096 seed=11 experiment_id=94 &
export CUDA_VISIBLE_DEVICES=2; python train.py task=metaworld_plate-slide-back-v2 disentangled_version=V1 latent_dim=4096 seed=12 experiment_id=94 &
export CUDA_VISIBLE_DEVICES=3; python train.py task=metaworld_plate-slide-back-side-v2 disentangled_version=original_drqv2 seed=10 experiment_id=95 &
export CUDA_VISIBLE_DEVICES=3; python train.py task=metaworld_plate-slide-back-side-v2 disentangled_version=original_drqv2 seed=11 experiment_id=95 &
export CUDA_VISIBLE_DEVICES=3; python train.py task=metaworld_plate-slide-back-side-v2 disentangled_version=original_drqv2 seed=12 experiment_id=95 &
export CUDA_VISIBLE_DEVICES=4; python train.py task=metaworld_plate-slide-back-side-v2 disentangled_version=V1 latent_dim=4096 seed=10 experiment_id=96 &
export CUDA_VISIBLE_DEVICES=4; python train.py task=metaworld_plate-slide-back-side-v2 disentangled_version=V1 latent_dim=4096 seed=11 experiment_id=96 &
export CUDA_VISIBLE_DEVICES=4; python train.py task=metaworld_plate-slide-back-side-v2 disentangled_version=V1 latent_dim=4096 seed=12 experiment_id=96 &
export CUDA_VISIBLE_DEVICES=5; python train.py task=metaworld_peg-unplug-side-v2 disentangled_version=original_drqv2 seed=10 experiment_id=97 &
export CUDA_VISIBLE_DEVICES=5; python train.py task=metaworld_peg-unplug-side-v2 disentangled_version=original_drqv2 seed=11 experiment_id=97 &
export CUDA_VISIBLE_DEVICES=5; python train.py task=metaworld_peg-unplug-side-v2 disentangled_version=original_drqv2 seed=12 experiment_id=97 &
export CUDA_VISIBLE_DEVICES=6; python train.py task=metaworld_peg-unplug-side-v2 disentangled_version=V1 latent_dim=4096 seed=10 experiment_id=98 &
export CUDA_VISIBLE_DEVICES=6; python train.py task=metaworld_peg-unplug-side-v2 disentangled_version=V1 latent_dim=4096 seed=11 experiment_id=98 &
export CUDA_VISIBLE_DEVICES=6; python train.py task=metaworld_peg-unplug-side-v2 disentangled_version=V1 latent_dim=4096 seed=12 experiment_id=98 &
export CUDA_VISIBLE_DEVICES=7; python train.py task=metaworld_soccer-v2 disentangled_version=original_drqv2 seed=10 experiment_id=99 &
export CUDA_VISIBLE_DEVICES=7; python train.py task=metaworld_soccer-v2 disentangled_version=original_drqv2 seed=11 experiment_id=99 &
export CUDA_VISIBLE_DEVICES=7; python train.py task=metaworld_soccer-v2 disentangled_version=original_drqv2 seed=12 experiment_id=99 &

wait
