#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --time=48:00:00
#SBATCH --mem=948G
#SBATCH --gres=gpu:8
#SBATCH --partition=abhinavlong
#SBATCH --nodelist=grogu-2-6
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/2_6.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/2_6.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000
export pretrained_path=/home/sbahl2/research/DisentangledRep4RL/exp_local/119_13_2023.01.04_16:50:23/snapshot.pt


export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=10 experiment_id=435 latent_dim=4096 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=11 experiment_id=435 latent_dim=4096 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=12 experiment_id=435 latent_dim=4096 &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_peg-insert-side-v2 agent=drqv2 seed=10 experiment_id=436 pretrain.path=${pretrained_path} agent.use_pool_encoder=false &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_peg-insert-side-v2 agent=drqv2 seed=11 experiment_id=436 pretrain.path=${pretrained_path} agent.use_pool_encoder=false &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_peg-insert-side-v2 agent=drqv2 seed=12 experiment_id=436 pretrain.path=${pretrained_path} agent.use_pool_encoder=false &

export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_lever-pull-v2 agent=drqv2 seed=10 experiment_id=437 pretrain.path=${pretrained_path} agent.use_pool_encoder=false &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_lever-pull-v2 agent=drqv2 seed=11 experiment_id=437 pretrain.path=${pretrained_path} agent.use_pool_encoder=false &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_lever-pull-v2 agent=drqv2 seed=12 experiment_id=437 pretrain.path=${pretrained_path} agent.use_pool_encoder=false &

export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_handle-pull-v2 agent=drqv2 seed=10 experiment_id=438 pretrain.path=${pretrained_path} agent.use_pool_encoder=false &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_handle-pull-v2 agent=drqv2 seed=11 experiment_id=438 pretrain.path=${pretrained_path} agent.use_pool_encoder=false &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_handle-pull-v2 agent=drqv2 seed=12 experiment_id=438 pretrain.path=${pretrained_path} agent.use_pool_encoder=false &

export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_soccer-v2 agent=drqv2 seed=10 experiment_id=439 pretrain.path=${pretrained_path} agent.use_pool_encoder=false &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_soccer-v2 agent=drqv2 seed=11 experiment_id=439 pretrain.path=${pretrained_path} agent.use_pool_encoder=false &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_soccer-v2 agent=drqv2 seed=12 experiment_id=439 pretrain.path=${pretrained_path} agent.use_pool_encoder=false &

export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_push-wall-v2 agent=drqv2 seed=10 experiment_id=440 pretrain.path=${pretrained_path} agent.use_pool_encoder=false &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_push-wall-v2 agent=drqv2 seed=11 experiment_id=440 pretrain.path=${pretrained_path} agent.use_pool_encoder=false &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_push-wall-v2 agent=drqv2 seed=12 experiment_id=440 pretrain.path=${pretrained_path} agent.use_pool_encoder=false &

export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_plate-slide-v2 agent=drqv2 seed=10 experiment_id=441 pretrain.path=${pretrained_path} agent.use_pool_encoder=false &
export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_plate-slide-v2 agent=drqv2 seed=11 experiment_id=441 pretrain.path=${pretrained_path} agent.use_pool_encoder=false &
export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_plate-slide-v2 agent=drqv2 seed=12 experiment_id=441 pretrain.path=${pretrained_path} agent.use_pool_encoder=false &

export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=metaworld_pick-out-of-hole-v2 agent=drqv2 seed=10 experiment_id=442 pretrain.path=${pretrained_path} agent.use_pool_encoder=false &
export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=metaworld_pick-out-of-hole-v2 agent=drqv2 seed=11 experiment_id=442 pretrain.path=${pretrained_path} agent.use_pool_encoder=false &
export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=metaworld_pick-out-of-hole-v2 agent=drqv2 seed=12 experiment_id=442 pretrain.path=${pretrained_path} agent.use_pool_encoder=false &


wait