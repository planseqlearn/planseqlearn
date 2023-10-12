#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --mem=400G
#SBATCH --gres=gpu:4
#SBATCH --partition=abhinavlong
#SBATCH --nodelist=grogu-2-3
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/2_3.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/2_3.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_door-close-v2 agent=drqv2AE seed=10 experiment_id=278 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_door-close-v2 agent=drqv2AE seed=11 experiment_id=278 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_door-close-v2 agent=drqv2AE seed=12 experiment_id=278 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_door-lock-v2 agent=drqv2AE seed=10 experiment_id=279 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_door-lock-v2 agent=drqv2AE seed=11 experiment_id=279 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_door-lock-v2 agent=drqv2AE seed=12 experiment_id=279 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &

export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_door-open-v2 agent=drqv2AE seed=10 experiment_id=280 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_door-open-v2 agent=drqv2AE seed=11 experiment_id=280 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_door-open-v2 agent=drqv2AE seed=12 experiment_id=280 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &

export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_door-unlock-v2 agent=drqv2AE seed=10 experiment_id=281 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_door-unlock-v2 agent=drqv2AE seed=11 experiment_id=281 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_door-unlock-v2 agent=drqv2AE seed=12 experiment_id=281 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &

wait
