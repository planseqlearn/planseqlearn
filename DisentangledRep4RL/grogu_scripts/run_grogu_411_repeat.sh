#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=50
#SBATCH --time=48:00:00
#SBATCH --mem=350G
#SBATCH --gres=gpu:2
#SBATCH --partition=abhinavlong
#SBATCH --nodelist=grogu-1-34
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/1_34.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/1_34.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000


export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-microwave-v0 agent=drqv2AE seed=10 experiment_id=416 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=random &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-microwave-v0 agent=drqv2AE seed=11 experiment_id=416 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=random &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 agent=V1 seed=10 experiment_id=425 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 latent_dim=4096 camera_name=random &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 agent=V1 seed=11 experiment_id=425 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 latent_dim=4096 camera_name=random &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 agent=V1 seed=12 experiment_id=425 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 latent_dim=4096 camera_name=random &

wait