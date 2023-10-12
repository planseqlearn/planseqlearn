#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=56
#SBATCH --time=48:00:00
#SBATCH --mem=448G
#SBATCH --gres=gpu:4
#SBATCH --partition=abhinavlong
#SBATCH --nodelist=grogu-1-24
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/1_24.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/1_24.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_door-lock-v2 agent=drqv2AE seed=10 experiment_id=356 latent_dim=4096 agent.reconstruction_loss_coeff=0.02 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_door-lock-v2 agent=drqv2AE seed=11 experiment_id=356 latent_dim=4096 agent.reconstruction_loss_coeff=0.02 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_door-lock-v2 agent=drqv2AE seed=12 experiment_id=356 latent_dim=4096 agent.reconstruction_loss_coeff=0.02 &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_door-open-v2 agent=drqv2AE seed=10 experiment_id=357 latent_dim=4096 agent.reconstruction_loss_coeff=0.02 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_door-open-v2 agent=drqv2AE seed=11 experiment_id=357 latent_dim=4096 agent.reconstruction_loss_coeff=0.02 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_door-open-v2 agent=drqv2AE seed=12 experiment_id=357 latent_dim=4096 agent.reconstruction_loss_coeff=0.02 &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_door-unlock-v2 agent=drqv2AE seed=10 experiment_id=358 latent_dim=4096 agent.reconstruction_loss_coeff=0.02 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_door-unlock-v2 agent=drqv2AE seed=11 experiment_id=358 latent_dim=4096 agent.reconstruction_loss_coeff=0.02 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_door-unlock-v2 agent=drqv2AE seed=12 experiment_id=358 latent_dim=4096 agent.reconstruction_loss_coeff=0.02 &

export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_hammer-v2 agent=drqv2AE seed=10 experiment_id=359 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_hammer-v2 agent=drqv2AE seed=11 experiment_id=359 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_hammer-v2 agent=drqv2AE seed=12 experiment_id=359 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &



wait