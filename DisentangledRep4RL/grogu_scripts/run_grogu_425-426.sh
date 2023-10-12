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

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_mt3-customized agent=drqv2AE seed=18 experiment_id=425 latent_dim=512 agent.reconstruction_loss_coeff=0.02 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_mt3-customized agent=drqv2AE seed=19 experiment_id=425 latent_dim=512 agent.reconstruction_loss_coeff=0.02 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_mt3-customized agent=drqv2AE seed=120 experiment_id=425 latent_dim=512 agent.reconstruction_loss_coeff=0.02 &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_mt3-customized agent=drqv2AE seed=18 experiment_id=426 latent_dim=512 agent.reconstruction_loss_coeff=0.002 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_mt3-customized agent=drqv2AE seed=19 experiment_id=426 latent_dim=512 agent.reconstruction_loss_coeff=0.002 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_mt3-customized agent=drqv2AE seed=20 experiment_id=426 latent_dim=512 agent.reconstruction_loss_coeff=0.002 &

export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_mt3-customized agent=drqv2AE seed=18 experiment_id=372 latent_dim=512 agent.reconstruction_loss_coeff=0.2 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_mt3-customized agent=drqv2AE seed=19 experiment_id=372 latent_dim=512 agent.reconstruction_loss_coeff=0.2 &

wait