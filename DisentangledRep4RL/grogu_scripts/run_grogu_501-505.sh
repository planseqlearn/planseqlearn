#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=70
#SBATCH --time=48:00:00
#SBATCH --mem=348G
#SBATCH --gres=gpu:7
#SBATCH --partition=deepaklong
#SBATCH --nodelist=grogu-0-19
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/0_19.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/0_19.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000


export pretrained_path_ae=/home/sbahl2/research/DisentangledRep4RL/exp_local/374_11_2023.01.17_19:37:09/latest.pt
export pretrained_path_dq=/home/sbahl2/research/DisentangledRep4RL/exp_local/119_13_2023.01.04_16:50:23/snapshot.pt
export pretrained_path_v1=/home/sbahl2/research/DisentangledRep4RL/exp_local/188_10_2023.01.12_22:16:57/snapshot.pt

export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 agent=drqv2AE seed=10 experiment_id=4140 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=random &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 agent=drqv2AE seed=11 experiment_id=4140 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=random &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 agent=drqv2AE seed=12 experiment_id=4140 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=random &

export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=drqv2AE seed=10 experiment_id=4150 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=random &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=drqv2AE seed=11 experiment_id=4150 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=random &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=drqv2AE seed=12 experiment_id=4150 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=random &

wait

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_door-lock-v2 agent=drqv2AE seed=10 experiment_id=279 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_door-lock-v2 agent=drqv2AE seed=11 experiment_id=279 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_door-lock-v2 agent=drqv2AE seed=12 experiment_id=279 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &

export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_door-open-v2 agent=drqv2AE seed=10 experiment_id=280 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_door-open-v2 agent=drqv2AE seed=11 experiment_id=280 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_door-open-v2 agent=drqv2AE seed=12 experiment_id=280 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &