#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --mem=180G
#SBATCH --gres=gpu:3
#SBATCH --partition=deepaklong
#SBATCH --nodelist=grogu-1-19
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/1_19.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/1_19.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_mt10 agent=V2 seed=10 experiment_id=182 agent.mask_loss_coeff=2.5e-1 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_mt10 agent=V2 seed=11 experiment_id=182 agent.mask_loss_coeff=2.5e-1 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_mt10 agent=V2 seed=12 experiment_id=182 agent.mask_loss_coeff=2.5e-1 &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_mt10 agent=V3 seed=10 experiment_id=183 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_mt10 agent=V3 seed=11 experiment_id=183 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_mt10 agent=V3 seed=12 experiment_id=183 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 &

wait
