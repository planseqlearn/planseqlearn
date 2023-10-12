#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --time=48:00:00
#SBATCH --mem=448G
#SBATCH --gres=gpu:8
#SBATCH --partition=deepaklong
#SBATCH --nodelist=grogu-1-19
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/1_19.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/1_19.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

# Recon to mask: 4
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=10 experiment_id=157 agent.mask_loss_coeff=4.9505e-6 agent.reconstruction_loss_coeff=1.9802e-5 agent.detach_critic=True &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=11 experiment_id=157 agent.mask_loss_coeff=4.9505e-6 agent.reconstruction_loss_coeff=1.9802e-5 agent.detach_critic=True &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=12 experiment_id=157 agent.mask_loss_coeff=4.9505e-6 agent.reconstruction_loss_coeff=1.9802e-5 agent.detach_critic=True &

# Recon to mask: 40
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=10 experiment_id=158 agent.mask_loss_coeff=4.5455e-6 agent.reconstruction_loss_coeff=1.8182e-4 agent.detach_critic=True &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=11 experiment_id=158 agent.mask_loss_coeff=4.5455e-6 agent.reconstruction_loss_coeff=1.8182e-4 agent.detach_critic=True &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=12 experiment_id=158 agent.mask_loss_coeff=4.5455e-6 agent.reconstruction_loss_coeff=1.8182e-4 agent.detach_critic=True &

# Recon to mask: 400
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=10 experiment_id=159 agent.detach_critic=True &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=11 experiment_id=159 agent.detach_critic=True &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=12 experiment_id=159 agent.detach_critic=True &

export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=10 experiment_id=160 &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=11 experiment_id=160 &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=12 experiment_id=160 &

export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=10 experiment_id=161 agent.mask_loss_coeff=2.5e-05 agent.reconstruction_loss_coeff=1e-2 &
export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=11 experiment_id=161 agent.mask_loss_coeff=2.5e-05 agent.reconstruction_loss_coeff=1e-2 &
export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=12 experiment_id=161 agent.mask_loss_coeff=2.5e-05 agent.reconstruction_loss_coeff=1e-2 &

wait
