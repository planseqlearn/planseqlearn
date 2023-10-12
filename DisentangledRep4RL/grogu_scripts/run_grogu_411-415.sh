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


export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=drqv2AE seed=10 experiment_id=411 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=drqv2AE seed=11 experiment_id=411 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=drqv2AE seed=12 experiment_id=411 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=fixed &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=kitchen_kitchen-microwave-v0 agent=drqv2AE seed=10 experiment_id=412 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=kitchen_kitchen-microwave-v0 agent=drqv2AE seed=11 experiment_id=412 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=kitchen_kitchen-microwave-v0 agent=drqv2AE seed=12 experiment_id=412 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=fixed &

export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 agent=drqv2AE seed=10 experiment_id=413 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 agent=drqv2AE seed=11 experiment_id=413 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 agent=drqv2AE seed=12 experiment_id=413 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=fixed &

# DrQv2AE, random camera
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 agent=drqv2AE seed=10 experiment_id=414 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=random &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 agent=drqv2AE seed=11 experiment_id=414 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=random &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 agent=drqv2AE seed=12 experiment_id=414 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=random &

export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=drqv2AE seed=10 experiment_id=415 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=random &
export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=drqv2AE seed=11 experiment_id=415 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=random &
export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=drqv2AE seed=12 experiment_id=415 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=random &


wait