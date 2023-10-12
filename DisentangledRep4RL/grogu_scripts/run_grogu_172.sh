#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --partition=deepaklong
#SBATCH --nodelist=grogu-1-40
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/1_40.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/1_40.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export pretrained_path=/home/sbahl2/research/DisentangledRep4RL/pretrained_encoders/158_11_encoder.pt

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=10 experiment_id=172 agent.mask_loss_coeff=2.5 agent.reconstruction_loss_coeff=10 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=11 experiment_id=172 agent.mask_loss_coeff=2.5 agent.reconstruction_loss_coeff=10 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=12 experiment_id=172 agent.mask_loss_coeff=2.5 agent.reconstruction_loss_coeff=10 &
wait
