#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --time=48:00:00
#SBATCH --mem=120G
#SBATCH --gres=gpu:2
#SBATCH --partition=abhinavlong
#SBATCH --nodelist=grogu-1-9
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/1_9.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/1_9.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_reach-wall-v2 agent=V1 latent_dim=512 seed=10 experiment_id=248 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_shelf-place-v2 agent=V1 latent_dim=512 seed=10 experiment_id=249 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_sweep-into-v2 agent=V1 latent_dim=512 seed=10 experiment_id=250 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_sweep-v2 agent=V1 latent_dim=512 seed=10 experiment_id=251 &

wait
