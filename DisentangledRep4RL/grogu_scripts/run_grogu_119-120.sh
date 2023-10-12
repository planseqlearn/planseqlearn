#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=60
#SBATCH --time=48:00:00
#SBATCH --mem=350G
#SBATCH --gres=gpu:5
#SBATCH --partition=deepaklong
#SBATCH --nodelist=grogu-1-19
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/1_19.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/1_19.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_mt10 agent=drqv2 seed=10 experiment_id=119 save_snapshot=true &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_mt10 agent=drqv2 seed=11 experiment_id=119 save_snapshot=true &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_mt10 agent=drqv2 seed=12 experiment_id=119 save_snapshot=true &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_mt10 agent=drqv2 seed=13 experiment_id=119 save_snapshot=true &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_mt10 agent=drqv2 seed=14 experiment_id=119 save_snapshot=true &

export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=10 experiment_id=120 save_snapshot=true &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=11 experiment_id=120 save_snapshot=true &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=12 experiment_id=120 save_snapshot=true &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=13 experiment_id=120 save_snapshot=true &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=14 experiment_id=120 save_snapshot=true &

wait
