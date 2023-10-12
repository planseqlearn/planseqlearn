#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --time=48:00:00
#SBATCH --mem=948G
#SBATCH --gres=gpu:8
#SBATCH --partition=deepaklong
#SBATCH --nodelist=grogu-1-3
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/1_3.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/1_3.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_bin-picking-v2 seed=10 experiment_id=80000 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_bin-picking-v2 seed=11 experiment_id=80000 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_bin-picking-v2 seed=12 experiment_id=80000 camera_name=corner agent=V1_random_mask &

# V1_random_mask, Adroit
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=adroit_hammer-human-v1 seed=10 experiment_id=80030 camera_name=fixed agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=adroit_hammer-human-v1 seed=11 experiment_id=80030 camera_name=fixed agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=adroit_hammer-human-v1 seed=12 experiment_id=80030 camera_name=fixed agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=adroit_pen-human-v1 seed=10 experiment_id=80040 camera_name=fixed agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=adroit_pen-human-v1 seed=11 experiment_id=80040 camera_name=fixed agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=adroit_pen-human-v1 seed=12 experiment_id=80040 camera_name=fixed agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=adroit_door-human-v1 seed=10 experiment_id=80050 camera_name=fixed agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=adroit_door-human-v1 seed=11 experiment_id=80050 camera_name=fixed agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=adroit_door-human-v1 seed=12 experiment_id=80050 camera_name=fixed agent=V1_random_mask &

# V1_random_mask, Kitchen, random camera
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 seed=10 experiment_id=80060 camera_name=random agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 seed=11 experiment_id=80060 camera_name=random agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 seed=12 experiment_id=80060 camera_name=random agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=kitchen_kitchen-light-v0 seed=10 experiment_id=80070 camera_name=random agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=kitchen_kitchen-light-v0 seed=11 experiment_id=80070 camera_name=random agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=kitchen_kitchen-light-v0 seed=12 experiment_id=80070 camera_name=random agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 seed=10 experiment_id=80080 camera_name=random agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 seed=11 experiment_id=80080 camera_name=random agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 seed=12 experiment_id=80080 camera_name=random agent=V1_random_mask &

# V1_random_mask, MT1
export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=metaworld_assembly-v2 seed=10 experiment_id=80090 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=metaworld_assembly-v2 seed=11 experiment_id=80090 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=metaworld_assembly-v2 seed=12 experiment_id=80090 camera_name=corner agent=V1_random_mask &


export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=metaworld_door-lock-v2 agent=drqv2AE seed=10 experiment_id=5070 latent_dim=4096 agent.reconstruction_loss_coeff=0.2 &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_door-lock-v2 agent=drqv2AE seed=11 experiment_id=5070 latent_dim=4096 agent.reconstruction_loss_coeff=0.2 &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_door-lock-v2 agent=drqv2AE seed=12 experiment_id=5070 latent_dim=4096 agent.reconstruction_loss_coeff=0.2 &

wait