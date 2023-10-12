#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --time=48:00:00
#SBATCH --mem=448G
#SBATCH --gres=gpu:8
#SBATCH --partition=abhinavlong
#SBATCH --nodelist=grogu-1-24
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/1_24.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/1_24.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export pretrained_path=/home/sbahl2/research/DisentangledRep4RL/exp_local/119_10_2023.01.04_16:50:24/snapshot.pt

# assembly, door lock, door close, hammer, dial turn
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_assembly-v2 agent=drqv2 seed=10 experiment_id=252 pretrain.path=${pretrained_path} &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_assembly-v2 agent=drqv2 seed=11 experiment_id=252 pretrain.path=${pretrained_path} &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_assembly-v2 agent=drqv2 seed=12 experiment_id=252 pretrain.path=${pretrained_path} &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_door-lock-v2 agent=drqv2 seed=10 experiment_id=253 pretrain.path=${pretrained_path} &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_door-lock-v2 agent=drqv2 seed=11 experiment_id=253 pretrain.path=${pretrained_path} &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_door-lock-v2 agent=drqv2 seed=12 experiment_id=253 pretrain.path=${pretrained_path} &

export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_door-close-v2 agent=drqv2 seed=10 experiment_id=254 pretrain.path=${pretrained_path} &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_door-close-v2 agent=drqv2 seed=11 experiment_id=254 pretrain.path=${pretrained_path} &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_door-close-v2 agent=drqv2 seed=12 experiment_id=254 pretrain.path=${pretrained_path} &

export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_hammer-v2 agent=drqv2 seed=10 experiment_id=255 pretrain.path=${pretrained_path} &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_hammer-v2 agent=drqv2 seed=11 experiment_id=255 pretrain.path=${pretrained_path} &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_hammer-v2 agent=drqv2 seed=12 experiment_id=255 pretrain.path=${pretrained_path} &

export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_dial-turn-v2 agent=drqv2 seed=10 experiment_id=256 pretrain.path=${pretrained_path} &
export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_dial-turn-v2 agent=drqv2 seed=11 experiment_id=256 pretrain.path=${pretrained_path} &
export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=metaworld_dial-turn-v2 agent=drqv2 seed=12 experiment_id=256 pretrain.path=${pretrained_path} &

wait