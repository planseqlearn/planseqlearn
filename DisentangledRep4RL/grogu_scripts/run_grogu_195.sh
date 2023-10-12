#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:2
#SBATCH --partition=abhinavlong
#SBATCH --nodelist=grogu-0-14
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/0_14.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/0_14.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-mixed-v0 agent=drqv2 seed=10 experiment_id=195 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-mixed-v0 agent=drqv2 seed=11 experiment_id=195 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=kitchen_kitchen-mixed-v0 agent=drqv2 seed=12 experiment_id=195 &

wait
