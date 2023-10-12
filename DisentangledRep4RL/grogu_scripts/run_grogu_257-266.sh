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

export pretrained_path_drqv2=/home/sbahl2/research/DisentangledRep4RL/exp_local/197_10_2023.01.13_17:46:22/snapshot.pt
export pretrained_path_V1=/home/sbahl2/research/DisentangledRep4RL/exp_local/198_10_2023.01.13_17:48:15/snapshot.pt

# button-press-topdown-wall, door open, door unlock, door lock, button-press-wall, assembly, hammer, handle-pull, door close, coffee button


# bin picking, box close, basketball, coffee pull, button press topdown
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_bin-picking-v2 agent=drqv2 seed=10 experiment_id=257 pretrain.path=${pretrained_path_drqv2} &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_bin-picking-v2 agent=drqv2 seed=11 experiment_id=257 pretrain.path=${pretrained_path_drqv2} &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_bin-picking-v2 agent=drqv2 seed=12 experiment_id=257 pretrain.path=${pretrained_path_drqv2} &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_box-close-v2 agent=drqv2 seed=10 experiment_id=258 pretrain.path=${pretrained_path_drqv2} &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_box-close-v2 agent=drqv2 seed=11 experiment_id=258 pretrain.path=${pretrained_path_drqv2} &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_box-close-v2 agent=drqv2 seed=12 experiment_id=258 pretrain.path=${pretrained_path_drqv2} &

export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_basketball-v2 agent=drqv2 seed=10 experiment_id=259 pretrain.path=${pretrained_path_drqv2} &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_basketball-v2 agent=drqv2 seed=11 experiment_id=259 pretrain.path=${pretrained_path_drqv2} &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_basketball-v2 agent=drqv2 seed=12 experiment_id=259 pretrain.path=${pretrained_path_drqv2} &

export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_coffee-pull-v2 agent=drqv2 seed=10 experiment_id=260 pretrain.path=${pretrained_path_drqv2} &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_coffee-pull-v2 agent=drqv2 seed=11 experiment_id=260 pretrain.path=${pretrained_path_drqv2} &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_coffee-pull-v2 agent=drqv2 seed=12 experiment_id=260 pretrain.path=${pretrained_path_drqv2} &

export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_button-press-topdown-v2 agent=drqv2 seed=10 experiment_id=261 pretrain.path=${pretrained_path_drqv2} &
export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_button-press-topdown-v2 agent=drqv2 seed=11 experiment_id=261 pretrain.path=${pretrained_path_drqv2} &
export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=metaworld_button-press-topdown-v2 agent=drqv2 seed=12 experiment_id=261 pretrain.path=${pretrained_path_drqv2} &


export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_bin-picking-v2 agent=V1 seed=10 experiment_id=262 pretrain.path=${pretrained_path_V1} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_bin-picking-v2 agent=V1 seed=11 experiment_id=262 pretrain.path=${pretrained_path_V1} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_bin-picking-v2 agent=V1 seed=12 experiment_id=262 pretrain.path=${pretrained_path_V1} latent_dim=512 &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=10 experiment_id=263 pretrain.path=${pretrained_path_V1} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=11 experiment_id=263 pretrain.path=${pretrained_path_V1} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=12 experiment_id=263 pretrain.path=${pretrained_path_V1} latent_dim=512 &

export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_basketball-v2 agent=V1 seed=10 experiment_id=264 pretrain.path=${pretrained_path_V1} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_basketball-v2 agent=V1 seed=11 experiment_id=264 pretrain.path=${pretrained_path_V1} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_basketball-v2 agent=V1 seed=12 experiment_id=264 pretrain.path=${pretrained_path_V1} latent_dim=512 &

export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_coffee-pull-v2 agent=V1 seed=10 experiment_id=265 pretrain.path=${pretrained_path_V1} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_coffee-pull-v2 agent=V1 seed=11 experiment_id=265 pretrain.path=${pretrained_path_V1} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_coffee-pull-v2 agent=V1 seed=12 experiment_id=265 pretrain.path=${pretrained_path_V1} latent_dim=512 &

export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_button-press-topdown-v2 agent=V1 seed=10 experiment_id=266 pretrain.path=${pretrained_path_V1} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_button-press-topdown-v2 agent=V1 seed=11 experiment_id=266 pretrain.path=${pretrained_path_V1} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=metaworld_button-press-topdown-v2 agent=V1 seed=12 experiment_id=266 pretrain.path=${pretrained_path_V1} latent_dim=512 &

wait
