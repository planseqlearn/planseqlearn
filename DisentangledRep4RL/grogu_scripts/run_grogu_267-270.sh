#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --mem=400G
#SBATCH --gres=gpu:4
#SBATCH --partition=abhinavlong
#SBATCH --nodelist=grogu-2-6
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/2_6.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/2_6.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=10 experiment_id=267 latent_dim=4096 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=11 experiment_id=267 latent_dim=4096 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=12 experiment_id=267 latent_dim=4096 &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=adroit_pen-human-v1 agent=drqv2AE seed=10 experiment_id=268 latent_dim=4096 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=adroit_pen-human-v1 agent=drqv2AE seed=11 experiment_id=268 latent_dim=4096 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=adroit_pen-human-v1 agent=drqv2AE seed=12 experiment_id=268 latent_dim=4096 &

export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=adroit_relocate-human-v1 agent=drqv2AE seed=10 experiment_id=269 latent_dim=4096 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=adroit_relocate-human-v1 agent=drqv2AE seed=11 experiment_id=269 latent_dim=4096 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=adroit_relocate-human-v1 agent=drqv2AE seed=12 experiment_id=269 latent_dim=4096 &

export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=adroit_door-human-v1 agent=drqv2AE seed=10 experiment_id=270 latent_dim=4096 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=adroit_door-human-v1 agent=drqv2AE seed=11 experiment_id=270 latent_dim=4096 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=adroit_door-human-v1 agent=drqv2AE seed=12 experiment_id=270 latent_dim=4096 &

wait
