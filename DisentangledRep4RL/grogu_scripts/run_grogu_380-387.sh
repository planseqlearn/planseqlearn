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
export pretrained_path=/home/sbahl2/research/DisentangledRep4RL/exp_local/188_10_2023.01.12_22:16:57/snapshot.pt

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_button-press-topdown-wall-v2 agent=V1 seed=10 experiment_id=380 pretrain.path=${pretrained_path} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_button-press-topdown-wall-v2 agent=V1 seed=11 experiment_id=380 pretrain.path=${pretrained_path} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_button-press-topdown-wall-v2 agent=V1 seed=12 experiment_id=380 pretrain.path=${pretrained_path} latent_dim=512 &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_button-press-topdown-v2 agent=V1 seed=10 experiment_id=381 pretrain.path=${pretrained_path} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_button-press-topdown-v2 agent=V1 seed=11 experiment_id=381 pretrain.path=${pretrained_path} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_button-press-topdown-v2 agent=V1 seed=12 experiment_id=381 pretrain.path=${pretrained_path} latent_dim=512 &

export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_door-unlock-v2 agent=V1 seed=10 experiment_id=382 pretrain.path=${pretrained_path} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_door-unlock-v2 agent=V1 seed=11 experiment_id=382 pretrain.path=${pretrained_path} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_door-unlock-v2 agent=V1 seed=12 experiment_id=382 pretrain.path=${pretrained_path} latent_dim=512 &

export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_bin-picking-v2 agent=V1 seed=10 experiment_id=383 pretrain.path=${pretrained_path} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_bin-picking-v2 agent=V1 seed=11 experiment_id=383 pretrain.path=${pretrained_path} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_bin-picking-v2 agent=V1 seed=12 experiment_id=383 pretrain.path=${pretrained_path} latent_dim=512 &

export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=10 experiment_id=384 pretrain.path=${pretrained_path} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=11 experiment_id=384 pretrain.path=${pretrained_path} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=12 experiment_id=384 pretrain.path=${pretrained_path} latent_dim=512 &

export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_coffee-pull-v2 agent=V1 seed=10 experiment_id=385 pretrain.path=${pretrained_path} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_coffee-pull-v2 agent=V1 seed=11 experiment_id=385 pretrain.path=${pretrained_path} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_coffee-pull-v2 agent=V1 seed=12 experiment_id=385 pretrain.path=${pretrained_path} latent_dim=512 &

export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_coffee-push-v2 agent=V1 seed=10 experiment_id=386 pretrain.path=${pretrained_path} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_coffee-push-v2 agent=V1 seed=11 experiment_id=386 pretrain.path=${pretrained_path} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_coffee-push-v2 agent=V1 seed=12 experiment_id=386 pretrain.path=${pretrained_path} latent_dim=512 &

export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=metaworld_coffee-button-v2 agent=V1 seed=10 experiment_id=387 pretrain.path=${pretrained_path} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=metaworld_coffee-button-v2 agent=V1 seed=11 experiment_id=387 pretrain.path=${pretrained_path} latent_dim=512 &
export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=metaworld_coffee-button-v2 agent=V1 seed=12 experiment_id=387 pretrain.path=${pretrained_path} latent_dim=512 &

wait