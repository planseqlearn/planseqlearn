#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=70
#SBATCH --time=48:00:00
#SBATCH --mem=390G
#SBATCH --gres=gpu:7
#SBATCH --partition=deepaklong
#SBATCH --nodelist=grogu-0-19
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/0_19.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/0_19.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export pretrained_path=/home/sbahl2/research/DisentangledRep4RL/exp_local/119_13_2023.01.04_16:50:23/snapshot.pt

# assembly, door lock, door close, hammer, dial turn
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_coffee-pull-v2 agent=drqv2 seed=10 experiment_id=432 pretrain.path=${pretrained_path} agent.use_pool_encoder=false &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_coffee-pull-v2 agent=drqv2 seed=11 experiment_id=432 pretrain.path=${pretrained_path} agent.use_pool_encoder=false &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_coffee-pull-v2 agent=drqv2 seed=12 experiment_id=432 pretrain.path=${pretrained_path} agent.use_pool_encoder=false &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_coffee-push-v2 agent=drqv2 seed=10 experiment_id=433 pretrain.path=${pretrained_path} agent.use_pool_encoder=false &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_coffee-push-v2 agent=drqv2 seed=11 experiment_id=433 pretrain.path=${pretrained_path} agent.use_pool_encoder=false &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_coffee-push-v2 agent=drqv2 seed=12 experiment_id=433 pretrain.path=${pretrained_path} agent.use_pool_encoder=false &

export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_coffee-button-v2 agent=drqv2 seed=10 experiment_id=434 pretrain.path=${pretrained_path} agent.use_pool_encoder=false &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_coffee-button-v2 agent=drqv2 seed=11 experiment_id=434 pretrain.path=${pretrained_path} agent.use_pool_encoder=false &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_coffee-button-v2 agent=drqv2 seed=12 experiment_id=435 pretrain.path=${pretrained_path} agent.use_pool_encoder=false &

export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 agent=drqv2 seed=10 experiment_id=405 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 agent=drqv2 seed=11 experiment_id=405 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 agent=drqv2 seed=12 experiment_id=405 camera_name=fixed &

wait
