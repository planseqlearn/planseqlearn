#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --mem=358G
#SBATCH --gres=gpu:6
#SBATCH --partition=deepaklong
#SBATCH --nodelist=grogu-0-19
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/0_19.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/0_19.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

# 0.1x
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=10 experiment_id=138 agent.mask_loss_coeff=2.5e-07 agent.reconstruction_loss_coeff=1e-4 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=11 experiment_id=138 agent.mask_loss_coeff=2.5e-07 agent.reconstruction_loss_coeff=1e-4 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=12 experiment_id=138 agent.mask_loss_coeff=2.5e-07 agent.reconstruction_loss_coeff=1e-4 &

# Experiments 139-141
# Constraint: mask * 100000 + recon * 250 = 0.5
# recon = 400 * mask -> mask = 2.5e-06, recon = 1e-3

# recon = 4 * mask -> mask = 4.9505e-6, recon = 1.9802e-5
# recon = 40 * mask -> mask = 4.5455e-6, recon = 1.8182e-4
# recon = 4000 * mask -> mask = 4.5454e-7, recon= 1.8181e-3

# Recon to mask: 4
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=10 experiment_id=139 agent.mask_loss_coeff=4.9505e-6 agent.reconstruction_loss_coeff=1.9802e-5 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=11 experiment_id=139 agent.mask_loss_coeff=4.9505e-6 agent.reconstruction_loss_coeff=1.9802e-5 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=12 experiment_id=139 agent.mask_loss_coeff=4.9505e-6 agent.reconstruction_loss_coeff=1.9802e-5 &

# Recon to mask: 40
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=10 experiment_id=140 agent.mask_loss_coeff=4.5455e-6 agent.reconstruction_loss_coeff=1.8182e-4 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=11 experiment_id=140 agent.mask_loss_coeff=4.5455e-6 agent.reconstruction_loss_coeff=1.8182e-4 &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=12 experiment_id=140 agent.mask_loss_coeff=4.5455e-6 agent.reconstruction_loss_coeff=1.8182e-4 &

# Recon to mask: 4000
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=10 experiment_id=141 agent.mask_loss_coeff=4.5454e-7 agent.reconstruction_loss_coeff=1.8181e-3 &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=11 experiment_id=141 agent.mask_loss_coeff=4.5454e-7 agent.reconstruction_loss_coeff=1.8181e-3 &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=12 experiment_id=141 agent.mask_loss_coeff=4.5454e-7 agent.reconstruction_loss_coeff=1.8181e-3 &

wait
