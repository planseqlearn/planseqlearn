#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --time=48:00:00
#SBATCH --mem=120G
#SBATCH --gres=gpu:2
#SBATCH --partition=deepaklong
#SBATCH --nodelist=grogu-0-19
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/0_19.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/0_19.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_coffee-pull-v2 agent=V1 latent_dim=512 seed=10 experiment_id=220 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_coffee-push-v2 agent=V1 latent_dim=512 seed=10 experiment_id=221 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_dial-turn-v2 agent=V1 latent_dim=512 seed=10 experiment_id=222 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_disassemble-v2 agent=V1 latent_dim=512 seed=10 experiment_id=223 &

wait
