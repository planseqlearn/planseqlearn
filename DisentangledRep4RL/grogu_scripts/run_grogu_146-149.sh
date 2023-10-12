#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --mem=350G
#SBATCH --gres=gpu:6
#SBATCH --partition=deepaklong
#SBATCH --nodelist=grogu-1-40
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/1_40.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/1_40.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

# 40x
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=10 experiment_id=146 agent.mask_loss_coeff=1e-04 agent.reconstruction_loss_coeff=4e-2 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=11 experiment_id=146 agent.mask_loss_coeff=1e-04 agent.reconstruction_loss_coeff=4e-2 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=12 experiment_id=146 agent.mask_loss_coeff=1e-04 agent.reconstruction_loss_coeff=4e-2 &

# 60x
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=10 experiment_id=147 agent.mask_loss_coeff=1.5e-04 agent.reconstruction_loss_coeff=6e-2 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=11 experiment_id=147 agent.mask_loss_coeff=1.5e-04 agent.reconstruction_loss_coeff=6e-2 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=12 experiment_id=147 agent.mask_loss_coeff=1.5e-04 agent.reconstruction_loss_coeff=6e-2 &

# 5x
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=10 experiment_id=148 agent.mask_loss_coeff=1.25e-05 agent.reconstruction_loss_coeff=5e-3 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=11 experiment_id=148 agent.mask_loss_coeff=1.25e-05 agent.reconstruction_loss_coeff=5e-3 &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=12 experiment_id=148 agent.mask_loss_coeff=1.25e-05 agent.reconstruction_loss_coeff=5e-3 &

# 20x
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=10 experiment_id=149 agent.mask_loss_coeff=5e-05 agent.reconstruction_loss_coeff=2e-2 &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=11 experiment_id=149 agent.mask_loss_coeff=5e-05 agent.reconstruction_loss_coeff=2e-2 &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=12 experiment_id=149 agent.mask_loss_coeff=5e-05 agent.reconstruction_loss_coeff=2e-2 &

wait
