#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=48:00:00
#SBATCH --mem=448G
#SBATCH --gres=gpu:4
#SBATCH --partition=deepaklong
#SBATCH --nodelist=grogu-1-40
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/1_40.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/1_40.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export pretrained_path_ae=/home/sbahl2/research/DisentangledRep4RL/exp_local/374_11_2023.01.17_19:37:09/latest.pt
export pretrained_path_dq=/home/sbahl2/research/DisentangledRep4RL/exp_local/119_13_2023.01.04_16:50:23/snapshot.pt
export pretrained_path_v1=/home/sbahl2/research/DisentangledRep4RL/exp_local/188_10_2023.01.12_22:16:57/snapshot.pt

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_peg-insert-side-v2 agent=V1 seed=10 experiment_id=484 pretrain.path=${pretrained_path_v1} latent_dim=512 agent.mask_loss_coeff=2.5e-4 agent.reconstruction_loss_coeff=1e-3 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_peg-insert-side-v2 agent=V1 seed=11 experiment_id=484 pretrain.path=${pretrained_path_v1} latent_dim=512 agent.mask_loss_coeff=2.5e-4 agent.reconstruction_loss_coeff=1e-3 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_peg-insert-side-v2 agent=V1 seed=12 experiment_id=484 pretrain.path=${pretrained_path_v1} latent_dim=512 agent.mask_loss_coeff=2.5e-4 agent.reconstruction_loss_coeff=1e-3 &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_soccer-v2 agent=drqv2AE seed=10 experiment_id=485 pretrain.path=${pretrained_path_ae} latent_dim=512 agent.reconstruction_loss_coeff=0.2&
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_soccer-v2 agent=drqv2AE seed=11 experiment_id=485 pretrain.path=${pretrained_path_ae} latent_dim=512 agent.reconstruction_loss_coeff=0.2&
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_soccer-v2 agent=drqv2AE seed=12 experiment_id=485 pretrain.path=${pretrained_path_ae} latent_dim=512 agent.reconstruction_loss_coeff=0.2&

export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_soccer-v2 agent=V1 seed=10 experiment_id=486 pretrain.path=${pretrained_path_v1} latent_dim=512 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_soccer-v2 agent=V1 seed=11 experiment_id=486 pretrain.path=${pretrained_path_v1} latent_dim=512 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_soccer-v2 agent=V1 seed=12 experiment_id=486 pretrain.path=${pretrained_path_v1} latent_dim=512 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &

export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_soccer-v2 agent=V1 seed=10 experiment_id=487 pretrain.path=${pretrained_path_v1} latent_dim=512 agent.mask_loss_coeff=2.5e-4 agent.reconstruction_loss_coeff=1e-3 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_soccer-v2 agent=V1 seed=11 experiment_id=487 pretrain.path=${pretrained_path_v1} latent_dim=512 agent.mask_loss_coeff=2.5e-4 agent.reconstruction_loss_coeff=1e-3 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_soccer-v2 agent=V1 seed=12 experiment_id=487 pretrain.path=${pretrained_path_v1} latent_dim=512 agent.mask_loss_coeff=2.5e-4 agent.reconstruction_loss_coeff=1e-3 &


wait