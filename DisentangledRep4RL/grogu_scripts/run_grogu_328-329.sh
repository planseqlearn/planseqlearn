#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=30
#SBATCH --time=48:00:00
#SBATCH --mem=168G
#SBATCH --gres=gpu:3
#SBATCH --partition=deepaklong
#SBATCH --nodelist=grogu-0-19
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/0_19_2.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/0_19_2.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=V1 seed=10 experiment_id=328 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 latent_dim=1024 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=V1 seed=11 experiment_id=328 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 latent_dim=1024 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=V1 seed=12 experiment_id=328 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 latent_dim=1024 &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=10 experiment_id=329 agent.reconstruction_loss_coeff=2e-1 latent_dim=4096 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=11 experiment_id=329 agent.reconstruction_loss_coeff=2e-1 latent_dim=4096 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=12 experiment_id=329 agent.reconstruction_loss_coeff=2e-1 latent_dim=4096 &

wait
