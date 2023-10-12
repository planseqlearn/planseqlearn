#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --time=48:00:00
#SBATCH --mem=800G
#SBATCH --gres=gpu:8
#SBATCH --partition=abhinavlong
#SBATCH --nodelist=grogu-2-3
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/2_3.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/2_3.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_mt3-customized agent=drqv2AE seed=15 experiment_id=393 latent_dim=512 agent.reconstruction_loss_coeff=0.02 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_mt3-customized agent=drqv2AE seed=16 experiment_id=393 latent_dim=512 agent.reconstruction_loss_coeff=0.02 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_mt3-customized agent=drqv2AE seed=17 experiment_id=393 latent_dim=512 agent.reconstruction_loss_coeff=0.02 &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_mt3-customized agent=drqv2AE seed=10 experiment_id=394 latent_dim=512 agent.reconstruction_loss_coeff=0.002 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_mt3-customized agent=drqv2AE seed=11 experiment_id=394 latent_dim=512 agent.reconstruction_loss_coeff=0.002 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_mt3-customized agent=drqv2AE seed=12 experiment_id=394 latent_dim=512 agent.reconstruction_loss_coeff=0.002 &

export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=10 experiment_id=395 latent_dim=4096 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=11 experiment_id=395 latent_dim=4096 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=12 experiment_id=395 latent_dim=4096 &

export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 agent=drqv2 seed=10 experiment_id=402 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 agent=drqv2 seed=11 experiment_id=402 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 agent=drqv2 seed=12 experiment_id=402 camera_name=fixed &

export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 agent=drqv2AE seed=10 experiment_id=410 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 agent=drqv2AE seed=11 experiment_id=410 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 agent=drqv2AE seed=12 experiment_id=410 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=fixed &

export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 agent=V1 seed=10 experiment_id=418 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 latent_dim=4096 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 agent=V1 seed=11 experiment_id=418 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 latent_dim=4096 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 agent=V1 seed=12 experiment_id=418 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 latent_dim=4096 camera_name=fixed &

export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=drqv2 seed=10 experiment_id=403 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=drqv2 seed=11 experiment_id=403 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=drqv2 seed=12 experiment_id=403 camera_name=fixed &

export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=kitchen_kitchen-microwave-v0 agent=drqv2 seed=10 experiment_id=404 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=kitchen_kitchen-microwave-v0 agent=drqv2 seed=11 experiment_id=404 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=kitchen_kitchen-microwave-v0 agent=drqv2 seed=12 experiment_id=404 camera_name=fixed &

wait