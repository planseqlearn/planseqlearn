#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --time=48:00:00
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --partition=deepaklong
#SBATCH --nodelist=grogu-0-19
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/0_19_2.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/0_19_2.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=10 experiment_id=142 agent.detach_critic=True &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_mt50 agent=V1 seed=10 experiment_id=143 agent.detach_critic=True &

wait
