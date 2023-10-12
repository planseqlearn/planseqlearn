#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=18
#SBATCH --time=48:00:00
#SBATCH --mem=90G
#SBATCH --gres=gpu:2
#SBATCH --partition=deepaklong
#SBATCH --nodelist=grogu-0-19
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/0_19.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/0_19.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_assembly-v2 agent=drqv2AE seed=10 experiment_id=273 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_assembly-v2 agent=drqv2AE seed=11 experiment_id=273 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_assembly-v2 agent=drqv2AE seed=12 experiment_id=273 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &

wait
