#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=256
#SBATCH --time=48:00:00
#SBATCH --mem=448G
#SBATCH --gres=gpu:4
#SBATCH --partition=deepaklong
#SBATCH --nodelist=grogu-0-24
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/0_24.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/0_24.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000


export pretrained_path=/home/sbahl2/research/DisentangledRep4RL/exp_local/374_11_2023.01.17_19:37:09/latest.pt



export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_door-lock-v2 agent=drqv2AE seed=10 experiment_id=448 pretrain.path=${pretrained_path} latent_dim=512 agent.reconstruction_loss_coeff=0.02&
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_door-lock-v2 agent=drqv2AE seed=11 experiment_id=448 pretrain.path=${pretrained_path} latent_dim=512 agent.reconstruction_loss_coeff=0.02&
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_door-lock-v2 agent=drqv2AE seed=12 experiment_id=448 pretrain.path=${pretrained_path} latent_dim=512 agent.reconstruction_loss_coeff=0.02&

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=20 experiment_id=475 latent_dim=4096 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=21 experiment_id=475 latent_dim=4096 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=22 experiment_id=475 latent_dim=4096 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=23 experiment_id=475 latent_dim=4096 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=24 experiment_id=475 latent_dim=4096 &
wait