#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=256
#SBATCH --time=48:00:00
#SBATCH --mem=448G
#SBATCH --gres=gpu:4
#SBATCH --partition=deepaklong
#SBATCH --nodelist=grogu-0-24
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/0_24.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/0_24.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=13 experiment_id=360 latent_dim=512 agent.reconstruction_loss_coeff=2 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=14 experiment_id=360 latent_dim=512 agent.reconstruction_loss_coeff=2 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=15 experiment_id=360 latent_dim=512 agent.reconstruction_loss_coeff=2 &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=13 experiment_id=361 latent_dim=4096 agent.reconstruction_loss_coeff=1 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=14 experiment_id=361 latent_dim=4096 agent.reconstruction_loss_coeff=1 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=15 experiment_id=361 latent_dim=4096 agent.reconstruction_loss_coeff=1 &

export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=13 experiment_id=362 latent_dim=4096 agent.reconstruction_loss_coeff=3 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=14 experiment_id=362 latent_dim=4096 agent.reconstruction_loss_coeff=3 &



wait