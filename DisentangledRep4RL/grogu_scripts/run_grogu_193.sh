#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=18
#SBATCH --time=48:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:3
#SBATCH --partition=deepaklong
#SBATCH --nodelist=grogu-0-24
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/0_24.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/0_24.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-complete-v0 agent=drqv2 seed=10 experiment_id=193 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=kitchen_kitchen-complete-v0 agent=drqv2 seed=11 experiment_id=193 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=kitchen_kitchen-complete-v0 agent=drqv2 seed=12 experiment_id=193 &

wait
