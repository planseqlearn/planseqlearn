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

export pretrained_path=/home/sbahl2/research/DisentangledRep4RL/exp_local/188_10_2023.01.12_22:16:57/snapshot.pt

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_lever-pull-v2 agent=V1 seed=10 experiment_id=443 pretrain.path=${pretrained_path} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_lever-pull-v2 agent=V1 seed=11 experiment_id=443 pretrain.path=${pretrained_path} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_lever-pull-v2 agent=V1 seed=12 experiment_id=443 pretrain.path=${pretrained_path} latent_dim=512 &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_handle-pull-v2 agent=V1 seed=10 experiment_id=444 pretrain.path=${pretrained_path} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_handle-pull-v2 agent=V1 seed=11 experiment_id=444 pretrain.path=${pretrained_path} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_handle-pull-v2 agent=V1 seed=12 experiment_id=444 pretrain.path=${pretrained_path} latent_dim=512 &

export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_soccer-v2 agent=V1 seed=10 experiment_id=445 pretrain.path=${pretrained_path} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_soccer-v2 agent=V1 seed=11 experiment_id=445 pretrain.path=${pretrained_path} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_soccer-v2 agent=V1 seed=12 experiment_id=445 pretrain.path=${pretrained_path} latent_dim=512 &

export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_plate-slide-v2 agent=V1 seed=10 experiment_id=446 pretrain.path=${pretrained_path} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_plate-slide-v2 agent=V1 seed=11 experiment_id=446 pretrain.path=${pretrained_path} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_plate-slide-v2 agent=V1 seed=12 experiment_id=446 pretrain.path=${pretrained_path} latent_dim=512 &

export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_pick-out-of-hole-v2 agent=V1 seed=10 experiment_id=447 pretrain.path=${pretrained_path} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_pick-out-of-hole-v2 agent=V1 seed=11 experiment_id=447 pretrain.path=${pretrained_path} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=metaworld_pick-out-of-hole-v2 agent=V1 seed=12 experiment_id=447 pretrain.path=${pretrained_path} latent_dim=512 &


wait