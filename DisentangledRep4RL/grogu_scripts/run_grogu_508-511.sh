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


export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_door-open-v2 agent=drqv2AE seed=20 experiment_id=508 latent_dim=4096 agent.reconstruction_loss_coeff=0.2 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_door-open-v2 agent=drqv2AE seed=21 experiment_id=508 latent_dim=4096 agent.reconstruction_loss_coeff=0.2 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_door-open-v2 agent=drqv2AE seed=22 experiment_id=508 latent_dim=4096 agent.reconstruction_loss_coeff=0.2 &

export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=V1 seed=10 experiment_id=510 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 latent_dim=4096 camera_name=random &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=V1 seed=11 experiment_id=510 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 latent_dim=4096 camera_name=random &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=V1 seed=12 experiment_id=510 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 latent_dim=4096 camera_name=random &

export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=V1 seed=10 experiment_id=511 agent.mask_loss_coeff=5e-2 agent.reconstruction_loss_coeff=5e-1 latent_dim=4096 camera_name=random &
export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=V1 seed=11 experiment_id=511 agent.mask_loss_coeff=5e-2 agent.reconstruction_loss_coeff=5e-1 latent_dim=4096 camera_name=random &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=V1 seed=12 experiment_id=511 agent.mask_loss_coeff=5e-2 agent.reconstruction_loss_coeff=5e-1 latent_dim=4096 camera_name=random &

export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=V1 seed=10 experiment_id=509 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 latent_dim=4096 camera_name=random &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=V1 seed=11 experiment_id=509 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 latent_dim=4096 camera_name=random &
export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=V1 seed=12 experiment_id=509 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 latent_dim=4096 camera_name=random &


wait


