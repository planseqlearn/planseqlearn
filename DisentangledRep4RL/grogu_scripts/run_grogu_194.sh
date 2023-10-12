#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --partition=abhinavlong
#SBATCH --nodelist=grogu-2-6
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/2_6.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/2_6.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-complete-v0 agent=V1 seed=10 experiment_id=194 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-complete-v0 agent=V1 seed=11 experiment_id=194 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-complete-v0 agent=V1 seed=12 experiment_id=194 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 &

wait
