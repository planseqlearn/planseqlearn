#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --mem=360G
#SBATCH --gres=gpu:6
#SBATCH --partition=deepaklong
#SBATCH --nodelist=grogu-0-19
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/0_19.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/0_19.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit-hammer-human-v1 agent=V1 seed=10 experiment_id=178 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=V1 seed=11 experiment_id=178 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=V1 seed=12 experiment_id=178 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=adroit_pen-human-v1 agent=V1 seed=10 experiment_id=179 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=adroit_pen-human-v1 agent=V1 seed=11 experiment_id=179 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=adroit_pen-human-v1 agent=V1 seed=12 experiment_id=179 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 &

export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=adroit_relocate-human-v1 agent=V1 seed=10 experiment_id=180 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=adroit_relocate-human-v1 agent=V1 seed=11 experiment_id=180 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=adroit_relocate-human-v1 agent=V1 seed=12 experiment_id=180 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 &

export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=adroit_door-human-v1 agent=V1 seed=10 experiment_id=181 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=adroit_door-human-v1 agent=V1 seed=11 experiment_id=181 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=adroit_door-human-v1 agent=V1 seed=12 experiment_id=181 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 &

wait
