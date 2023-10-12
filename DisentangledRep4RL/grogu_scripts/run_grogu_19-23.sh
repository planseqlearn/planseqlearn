#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --time=48:00:00
#SBATCH --mem=448G
#SBATCH --gres=gpu:8
#SBATCH --partition=deepaklong
#SBATCH --nodelist=grogu-1-19
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/1_19.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/1_19.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export CUDA_VISIBLE_DEVICES=0; python train.py task=metaworld_assembly-v2 disentangled_version=original_drqv2 seed=10 experiment_id=19 &
export CUDA_VISIBLE_DEVICES=0; python train.py task=metaworld_assembly-v2 disentangled_version=original_drqv2 seed=11 experiment_id=19 &
export CUDA_VISIBLE_DEVICES=1; python train.py task=metaworld_assembly-v2 disentangled_version=original_drqv2 seed=12 experiment_id=19 &
export CUDA_VISIBLE_DEVICES=1; python train.py task=metaworld_assembly-v2 disentangled_version=V1 latent_dim=4096 seed=10 experiment_id=20 &
export CUDA_VISIBLE_DEVICES=2; python train.py task=metaworld_assembly-v2 disentangled_version=V1 latent_dim=4096 seed=11 experiment_id=20 &
export CUDA_VISIBLE_DEVICES=2; python train.py task=metaworld_assembly-v2 disentangled_version=V1 latent_dim=4096 seed=12 experiment_id=20 &
export CUDA_VISIBLE_DEVICES=3; python train.py task=metaworld_basketball-v2 disentangled_version=original_drqv2 seed=10 experiment_id=21 &
export CUDA_VISIBLE_DEVICES=3; python train.py task=metaworld_basketball-v2 disentangled_version=original_drqv2 seed=11 experiment_id=21 &
export CUDA_VISIBLE_DEVICES=4; python train.py task=metaworld_basketball-v2 disentangled_version=original_drqv2 seed=12 experiment_id=21 &
export CUDA_VISIBLE_DEVICES=4; python train.py task=metaworld_basketball-v2 disentangled_version=V1 latent_dim=4096 seed=10 experiment_id=22 &
export CUDA_VISIBLE_DEVICES=5; python train.py task=metaworld_basketball-v2 disentangled_version=V1 latent_dim=4096 seed=11 experiment_id=22 &
export CUDA_VISIBLE_DEVICES=5; python train.py task=metaworld_basketball-v2 disentangled_version=V1 latent_dim=4096 seed=12 experiment_id=22 &
export CUDA_VISIBLE_DEVICES=6; python train.py task=metaworld_bin-picking-v2 disentangled_version=original_drqv2 seed=10 experiment_id=23 &
export CUDA_VISIBLE_DEVICES=6; python train.py task=metaworld_bin-picking-v2 disentangled_version=original_drqv2 seed=11 experiment_id=23 &
export CUDA_VISIBLE_DEVICES=7; python train.py task=metaworld_bin-picking-v2 disentangled_version=original_drqv2 seed=12 experiment_id=23 &

# export CUDA_VISIBLE_DEVICES=0; python train.py task=metaworld_assembly-v2 disentangled_version=V1 seed=10 experiment_id=19 &
# export CUDA_VISIBLE_DEVICES=0; python train.py task=metaworld_basketball-v2 disentangled_version=V1 seed=10 experiment_id=20 &
# export CUDA_VISIBLE_DEVICES=0; python train.py task=metaworld_bin-picking-v2 disentangled_version=V1 seed=10 experiment_id=21 &
# export CUDA_VISIBLE_DEVICES=1; python train.py task=metaworld_box-close-v2 disentangled_version=V1 seed=10 experiment_id=22 &
# export CUDA_VISIBLE_DEVICES=1; python train.py task=metaworld_button-press-topdown-v2 disentangled_version=V1 seed=10 experiment_id=23 &
# export CUDA_VISIBLE_DEVICES=1; python train.py task=metaworld_button-press-topdown-wall-v2 disentangled_version=V1 seed=10 experiment_id=24 &
# export CUDA_VISIBLE_DEVICES=2; python train.py task=metaworld_button-press-v2 disentangled_version=V1 seed=10 experiment_id=25 &
# export CUDA_VISIBLE_DEVICES=2; python train.py task=metaworld_button-press-wall-v2 disentangled_version=V1 seed=10 experiment_id=26 &
# export CUDA_VISIBLE_DEVICES=2; python train.py task=metaworld_coffee-button-v2 disentangled_version=V1 seed=10 experiment_id=27 &
# export CUDA_VISIBLE_DEVICES=3; python train.py task=metaworld_coffee-pull-v2 disentangled_version=V1 seed=10 experiment_id=28 &
# export CUDA_VISIBLE_DEVICES=3; python train.py task=metaworld_coffee-push-v2 disentangled_version=V1 seed=10 experiment_id=29 &
# export CUDA_VISIBLE_DEVICES=3; python train.py task=metaworld_dial-turn-v2 disentangled_version=V1 seed=10 experiment_id=30 &
# export CUDA_VISIBLE_DEVICES=4; python train.py task=metaworld_disassemble-v2 disentangled_version=V1 seed=10 experiment_id=31 &
# export CUDA_VISIBLE_DEVICES=4; python train.py task=metaworld_door-close-v2 disentangled_version=V1 seed=10 experiment_id=32 &
# export CUDA_VISIBLE_DEVICES=4; python train.py task=metaworld_door-lock-v2 disentangled_version=V1 seed=10 experiment_id=33 &
# export CUDA_VISIBLE_DEVICES=5; python train.py task=metaworld_door-open-v2 disentangled_version=V1 seed=10 experiment_id=34 &
# export CUDA_VISIBLE_DEVICES=5; python train.py task=metaworld_door-unlock-v2 disentangled_version=V1 seed=10 experiment_id=35 &
# export CUDA_VISIBLE_DEVICES=5; python train.py task=metaworld_hand-insert-v2 disentangled_version=V1 seed=10 experiment_id=36 &
# export CUDA_VISIBLE_DEVICES=6; python train.py task=metaworld_drawer-close-v2 disentangled_version=V1 seed=10 experiment_id=37 &
# export CUDA_VISIBLE_DEVICES=6; python train.py task=metaworld_drawer-open-v2 disentangled_version=V1 seed=10 experiment_id=38 &
# export CUDA_VISIBLE_DEVICES=6; python train.py task=metaworld_faucet-open-v2 disentangled_version=V1 seed=10 experiment_id=39 &
# export CUDA_VISIBLE_DEVICES=7; python train.py task=metaworld_faucet-close-v2 disentangled_version=V1 seed=10 experiment_id=40 &
# export CUDA_VISIBLE_DEVICES=7; python train.py task=metaworld_hammer-v2 disentangled_version=V1 seed=10 experiment_id=41 &
# export CUDA_VISIBLE_DEVICES=7; python train.py task=metaworld_handle-press-side-v2 disentangled_version=V1 seed=10 experiment_id=42 &

wait
