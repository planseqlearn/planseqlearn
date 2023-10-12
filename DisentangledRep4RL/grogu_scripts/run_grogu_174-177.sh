#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --mem=400G
#SBATCH --gres=gpu:4
#SBATCH --partition=deepaklong
#SBATCH --nodelist=grogu-1-40
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/1_40.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/1_40.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2 seed=10 experiment_id=174 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2 seed=11 experiment_id=174 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2 seed=12 experiment_id=174 &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=adroit_pen-human-v1 agent=drqv2 seed=10 experiment_id=175 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=adroit_pen-human-v1 agent=drqv2 seed=11 experiment_id=175 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=adroit_pen-human-v1 agent=drqv2 seed=12 experiment_id=175 &

export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=adroit_relocate-human-v1 agent=drqv2 seed=10 experiment_id=176 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=adroit_relocate-human-v1 agent=drqv2 seed=11 experiment_id=176 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=adroit_relocate-human-v1 agent=drqv2 seed=12 experiment_id=176 &

export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=adroit_door-human-v1 agent=drqv2 seed=10 experiment_id=177 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=adroit_door-human-v1 agent=drqv2 seed=11 experiment_id=177 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=adroit_door-human-v1 agent=drqv2 seed=12 experiment_id=177 &

wait
