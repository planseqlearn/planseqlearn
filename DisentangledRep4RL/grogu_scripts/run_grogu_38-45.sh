#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --time=48:00:00
#SBATCH --mem=900G
#SBATCH --gres=gpu:8
#SBATCH --partition=abhinavlong
#SBATCH --nodelist=grogu-2-6
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/2_6.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/2_6.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export CUDA_VISIBLE_DEVICES=0; python train.py task=metaworld_coffee-pull-v2 disentangled_version=V1 latent_dim=4096 seed=10 experiment_id=38 &
export CUDA_VISIBLE_DEVICES=0; python train.py task=metaworld_coffee-pull-v2 disentangled_version=V1 latent_dim=4096 seed=11 experiment_id=38 &
export CUDA_VISIBLE_DEVICES=0; python train.py task=metaworld_coffee-pull-v2 disentangled_version=V1 latent_dim=4096 seed=12 experiment_id=38 &

export CUDA_VISIBLE_DEVICES=1; python train.py task=metaworld_coffee-push-v2 disentangled_version=original_drqv2 seed=10 experiment_id=39 &
export CUDA_VISIBLE_DEVICES=1; python train.py task=metaworld_coffee-push-v2 disentangled_version=original_drqv2 seed=11 experiment_id=39 &
export CUDA_VISIBLE_DEVICES=1; python train.py task=metaworld_coffee-push-v2 disentangled_version=original_drqv2 seed=12 experiment_id=39 &

export CUDA_VISIBLE_DEVICES=2; python train.py task=metaworld_coffee-push-v2 disentangled_version=V1 latent_dim=4096 seed=10 experiment_id=40 &
export CUDA_VISIBLE_DEVICES=2; python train.py task=metaworld_coffee-push-v2 disentangled_version=V1 latent_dim=4096 seed=11 experiment_id=40 &
export CUDA_VISIBLE_DEVICES=2; python train.py task=metaworld_coffee-push-v2 disentangled_version=V1 latent_dim=4096 seed=12 experiment_id=40 &

export CUDA_VISIBLE_DEVICES=3; python train.py task=metaworld_dial-turn-v2 disentangled_version=original_drqv2 seed=10 experiment_id=41 &
export CUDA_VISIBLE_DEVICES=3; python train.py task=metaworld_dial-turn-v2 disentangled_version=original_drqv2 seed=11 experiment_id=41 &
export CUDA_VISIBLE_DEVICES=3; python train.py task=metaworld_dial-turn-v2 disentangled_version=original_drqv2 seed=12 experiment_id=41 &

export CUDA_VISIBLE_DEVICES=4; python train.py task=metaworld_dial-turn-v2 disentangled_version=V1 latent_dim=4096 seed=10 experiment_id=42 &
export CUDA_VISIBLE_DEVICES=4; python train.py task=metaworld_dial-turn-v2 disentangled_version=V1 latent_dim=4096 seed=11 experiment_id=42 &
export CUDA_VISIBLE_DEVICES=4; python train.py task=metaworld_dial-turn-v2 disentangled_version=V1 latent_dim=4096 seed=12 experiment_id=42 &

export CUDA_VISIBLE_DEVICES=5; python train.py task=metaworld_disassemble-v2 disentangled_version=original_drqv2 seed=10 experiment_id=43 &
export CUDA_VISIBLE_DEVICES=5; python train.py task=metaworld_disassemble-v2 disentangled_version=original_drqv2 seed=11 experiment_id=43 &
export CUDA_VISIBLE_DEVICES=5; python train.py task=metaworld_disassemble-v2 disentangled_version=original_drqv2 seed=12 experiment_id=43 &

export CUDA_VISIBLE_DEVICES=6; python train.py task=metaworld_disassemble-v2 disentangled_version=V1 latent_dim=4096 seed=10 experiment_id=44 &
export CUDA_VISIBLE_DEVICES=6; python train.py task=metaworld_disassemble-v2 disentangled_version=V1 latent_dim=4096 seed=11 experiment_id=44 &
export CUDA_VISIBLE_DEVICES=6; python train.py task=metaworld_disassemble-v2 disentangled_version=V1 latent_dim=4096 seed=12 experiment_id=44 &

export CUDA_VISIBLE_DEVICES=7; python train.py task=metaworld_door-close-v2 disentangled_version=original_drqv2 seed=10 experiment_id=45 &
export CUDA_VISIBLE_DEVICES=7; python train.py task=metaworld_door-close-v2 disentangled_version=original_drqv2 seed=11 experiment_id=45 &
export CUDA_VISIBLE_DEVICES=7; python train.py task=metaworld_door-close-v2 disentangled_version=original_drqv2 seed=12 experiment_id=45 &

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
