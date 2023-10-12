#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=36
#SBATCH --time=48:00:00
#SBATCH --mem=180G
#SBATCH --gres=gpu:3
#SBATCH --partition=deepaklong
#SBATCH --nodelist=grogu-0-24
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/0_24.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/0_24.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=10 experiment_id=144 agent.detach_decoders=True &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=11 experiment_id=144 agent.detach_decoders=True &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=12 experiment_id=144 agent.detach_decoders=True &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_mt50 agent=V1 seed=10 experiment_id=145 agent.detach_decoders=True &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_mt50 agent=V1 seed=11 experiment_id=145 agent.detach_decoders=True &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_mt50 agent=V1 seed=12 experiment_id=145 agent.detach_decoders=True &

wait
