#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=70
#SBATCH --time=48:00:00
#SBATCH --mem=390G
#SBATCH --gres=gpu:7
#SBATCH --partition=deepaklong
#SBATCH --nodelist=grogu-0-19
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/0_19.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/0_19.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=15 experiment_id=362 latent_dim=4096 agent.reconstruction_loss_coeff=3 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=13 experiment_id=363 latent_dim=4096 agent.reconstruction_loss_coeff=4 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=14 experiment_id=363 latent_dim=4096 agent.reconstruction_loss_coeff=4 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=15 experiment_id=363 latent_dim=4096 agent.reconstruction_loss_coeff=4 &

export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=adroit_pen-human-v1 agent=V1 seed=10 experiment_id=364 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=adroit_pen-human-v1 agent=V1 seed=11 experiment_id=364 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=adroit_pen-human-v1 agent=V1 seed=12 experiment_id=364 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 &

export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=adroit_pen-human-v1 agent=V1 seed=10 experiment_id=365 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=adroit_pen-human-v1 agent=V1 seed=11 experiment_id=365 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=adroit_pen-human-v1 agent=V1 seed=12 experiment_id=365 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &

export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=adroit_pen-human-v1 agent=V1 seed=10 experiment_id=366 agent.mask_loss_coeff=2.5e-4 agent.reconstruction_loss_coeff=1e-3 &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=adroit_pen-human-v1 agent=V1 seed=11 experiment_id=366 agent.mask_loss_coeff=2.5e-4 agent.reconstruction_loss_coeff=1e-3 &
export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=adroit_pen-human-v1 agent=V1 seed=12 experiment_id=366 agent.mask_loss_coeff=2.5e-4 agent.reconstruction_loss_coeff=1e-3 &

wait
