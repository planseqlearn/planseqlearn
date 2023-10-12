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

export CUDA_VISIBLE_DEVICES=0; python train.py task=metaworld_button-press-v2 disentangled_version=original_drqv2 seed=10 experiment_id=31 &
export CUDA_VISIBLE_DEVICES=0; python train.py task=metaworld_button-press-v2 disentangled_version=original_drqv2 seed=11 experiment_id=31 &
export CUDA_VISIBLE_DEVICES=0; python train.py task=metaworld_button-press-v2 disentangled_version=original_drqv2 seed=12 experiment_id=31 &

export CUDA_VISIBLE_DEVICES=1; python train.py task=metaworld_button-press-v2 disentangled_version=V1 latent_dim=4096 seed=10 experiment_id=32 &
export CUDA_VISIBLE_DEVICES=1; python train.py task=metaworld_button-press-v2 disentangled_version=V1 latent_dim=4096 seed=11 experiment_id=32 &
export CUDA_VISIBLE_DEVICES=1; python train.py task=metaworld_button-press-v2 disentangled_version=V1 latent_dim=4096 seed=12 experiment_id=32 &

export CUDA_VISIBLE_DEVICES=2; python train.py task=metaworld_button-press-wall-v2 disentangled_version=original_drqv2 seed=10 experiment_id=33 &
export CUDA_VISIBLE_DEVICES=2; python train.py task=metaworld_button-press-wall-v2 disentangled_version=original_drqv2 seed=11 experiment_id=33 &
export CUDA_VISIBLE_DEVICES=2; python train.py task=metaworld_button-press-wall-v2 disentangled_version=original_drqv2 seed=12 experiment_id=33 &

export CUDA_VISIBLE_DEVICES=3; python train.py task=metaworld_button-press-wall-v2 disentangled_version=V1 latent_dim=4096 seed=10 experiment_id=34 &
export CUDA_VISIBLE_DEVICES=3; python train.py task=metaworld_button-press-wall-v2 disentangled_version=V1 latent_dim=4096 seed=11 experiment_id=34 &
export CUDA_VISIBLE_DEVICES=3; python train.py task=metaworld_button-press-wall-v2 disentangled_version=V1 latent_dim=4096 seed=12 experiment_id=34 &

export CUDA_VISIBLE_DEVICES=4; python train.py task=metaworld_coffee-button-v2 disentangled_version=original_drqv2 seed=10 experiment_id=35 &
export CUDA_VISIBLE_DEVICES=4; python train.py task=metaworld_coffee-button-v2 disentangled_version=original_drqv2 seed=11 experiment_id=35 &
export CUDA_VISIBLE_DEVICES=4; python train.py task=metaworld_coffee-button-v2 disentangled_version=original_drqv2 seed=12 experiment_id=35 &

export CUDA_VISIBLE_DEVICES=5; python train.py task=metaworld_coffee-button-v2 disentangled_version=V1 latent_dim=4096 seed=10 experiment_id=36 &
export CUDA_VISIBLE_DEVICES=5; python train.py task=metaworld_coffee-button-v2 disentangled_version=V1 latent_dim=4096 seed=11 experiment_id=36 &
export CUDA_VISIBLE_DEVICES=5; python train.py task=metaworld_coffee-button-v2 disentangled_version=V1 latent_dim=4096 seed=12 experiment_id=36 &

export CUDA_VISIBLE_DEVICES=6; python train.py task=metaworld_coffee-pull-v2 disentangled_version=original_drqv2 seed=10 experiment_id=37 &
export CUDA_VISIBLE_DEVICES=6; python train.py task=metaworld_coffee-pull-v2 disentangled_version=original_drqv2 seed=11 experiment_id=37 &
export CUDA_VISIBLE_DEVICES=6; python train.py task=metaworld_coffee-pull-v2 disentangled_version=original_drqv2 seed=12 experiment_id=37 &

# export CUDA_VISIBLE_DEVICES=0; python train.py task=metaworld_peg-unplug-side-v2 disentangled_version=V1 seed=10 experiment_id=58 &
# export CUDA_VISIBLE_DEVICES=0; python train.py task=metaworld_soccer-v2 disentangled_version=V1 seed=10 experiment_id=59 &
# export CUDA_VISIBLE_DEVICES=0; python train.py task=metaworld_stick-push-v2 disentangled_version=V1 seed=10 experiment_id=60 &
# export CUDA_VISIBLE_DEVICES=1; python train.py task=metaworld_stick-pull-v2 disentangled_version=V1 seed=10 experiment_id=61 &
# export CUDA_VISIBLE_DEVICES=1; python train.py task=metaworld_push-wall-v2 disentangled_version=V1 seed=10 experiment_id=62 &
# export CUDA_VISIBLE_DEVICES=1; python train.py task=metaworld_reach-wall-v2 disentangled_version=V1 seed=10 experiment_id=63 &
# export CUDA_VISIBLE_DEVICES=2; python train.py task=metaworld_shelf-place-v2 disentangled_version=V1 seed=10 experiment_id=64 &
# export CUDA_VISIBLE_DEVICES=2; python train.py task=metaworld_sweep-into-v2 disentangled_version=V1 seed=10 experiment_id=65 &
# export CUDA_VISIBLE_DEVICES=2; python train.py task=metaworld_sweep-v2 disentangled_version=V1 seed=10 experiment_id=66 &
# export CUDA_VISIBLE_DEVICES=3; python train.py task=metaworld_window-open-v2 disentangled_version=V1 seed=10 experiment_id=67 &
# export CUDA_VISIBLE_DEVICES=3; python train.py task=metaworld_window-close-v2 disentangled_version=V1 seed=10 experiment_id=68 &
wait
