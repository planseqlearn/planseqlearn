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

export pretrained_path=/home/sbahl2/research/DisentangledRep4RL/exp_local/188_10_2023.01.12_22:16:57/snapshot.pt

# assembly, door lock, door close, hammer, dial turn
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_assembly-v2 agent=V1 seed=10 experiment_id=199 pretrain.path=${pretrained_path} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_assembly-v2 agent=V1 seed=11 experiment_id=199 pretrain.path=${pretrained_path} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_assembly-v2 agent=V1 seed=12 experiment_id=199 pretrain.path=${pretrained_path} latent_dim=512 &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_door-lock-v2 agent=V1 seed=10 experiment_id=200 pretrain.path=${pretrained_path} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_door-lock-v2 agent=V1 seed=11 experiment_id=200 pretrain.path=${pretrained_path} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_door-lock-v2 agent=V1 seed=12 experiment_id=200 pretrain.path=${pretrained_path} latent_dim=512 &

export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_door-close-v2 agent=V1 seed=10 experiment_id=201 pretrain.path=${pretrained_path} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_door-close-v2 agent=V1 seed=11 experiment_id=201 pretrain.path=${pretrained_path} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_door-close-v2 agent=V1 seed=12 experiment_id=201 pretrain.path=${pretrained_path} latent_dim=512 &

export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_hammer-v2 agent=V1 seed=10 experiment_id=202 pretrain.path=${pretrained_path} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_hammer-v2 agent=V1 seed=11 experiment_id=202 pretrain.path=${pretrained_path} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_hammer-v2 agent=V1 seed=12 experiment_id=202 pretrain.path=${pretrained_path} latent_dim=512 &

export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_dial-turn-v2 agent=V1 seed=10 experiment_id=203 pretrain.path=${pretrained_path} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_dial-turn-v2 agent=V1 seed=11 experiment_id=203 pretrain.path=${pretrained_path} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=metaworld_dial-turn-v2 agent=V1 seed=12 experiment_id=203 pretrain.path=${pretrained_path} latent_dim=512 &

wait
