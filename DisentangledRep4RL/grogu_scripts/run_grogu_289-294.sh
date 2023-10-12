#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=90
#SBATCH --time=48:00:00
#SBATCH --mem=540G
#SBATCH --gres=gpu:6
#SBATCH --partition=abhinavlong
#SBATCH --nodelist=grogu-2-6
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/2_6.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/2_6.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=V1 seed=10 experiment_id=289 agent.mask_loss_coeff=2.5e-4 agent.reconstruction_loss_coeff=1e-3 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=V1 seed=11 experiment_id=289 agent.mask_loss_coeff=2.5e-4 agent.reconstruction_loss_coeff=1e-3 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=V1 seed=12 experiment_id=289 agent.mask_loss_coeff=2.5e-4 agent.reconstruction_loss_coeff=1e-3 &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=V1 seed=10 experiment_id=290 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=V1 seed=11 experiment_id=290 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=V1 seed=12 experiment_id=290 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &

export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=V1 seed=10 experiment_id=291 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=V1 seed=11 experiment_id=291 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=V1 seed=12 experiment_id=291 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 &

export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=V1 seed=10 experiment_id=292 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 latent_dim=512 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=V1 seed=11 experiment_id=292 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 latent_dim=512 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=V1 seed=12 experiment_id=292 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 latent_dim=512 &

export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=V1 seed=10 experiment_id=293 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 latent_dim=1024 &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=V1 seed=11 experiment_id=293 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 latent_dim=1024 &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=V1 seed=12 experiment_id=293 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 latent_dim=1024 &

export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=V1 seed=10 experiment_id=294 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 latent_dim=2048 &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=V1 seed=11 experiment_id=294 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 latent_dim=2048 &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=V1 seed=12 experiment_id=294 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 latent_dim=2048 &

wait
