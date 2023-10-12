#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --time=48:00:00
#SBATCH --mem=250G
#SBATCH --gres=gpu:4
#SBATCH --partition=deepaklong
#SBATCH --nodelist=grogu-0-24
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/0_24.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/0_24.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_assembly-v2 agent=V1 latent_dim=512 seed=10 experiment_id=212 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_basketball-v2 agent=V1 latent_dim=512 seed=10 experiment_id=213 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_bin-picking-v2 agent=V1 latent_dim=512 seed=10 experiment_id=214 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 latent_dim=512 seed=10 experiment_id=215 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_button-press-topdown-wall-v2 agent=V1 latent_dim=512 seed=10 experiment_id=216 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_button-press-v2 agent=V1 latent_dim=512 seed=10 experiment_id=217 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_button-press-wall-v2 agent=V1 latent_dim=512 seed=10 experiment_id=218 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_coffee-button-v2 agent=V1 latent_dim=512 seed=10 experiment_id=219 &

wait
