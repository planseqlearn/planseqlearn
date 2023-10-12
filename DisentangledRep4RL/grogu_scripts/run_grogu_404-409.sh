#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --time=48:00:00
#SBATCH --mem=448G
#SBATCH --gres=gpu:8
#SBATCH --partition=deepaklong
#SBATCH --nodelist=grogu-0-19
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/0_19.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/0_19.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000


export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 agent=drqv2 seed=10 experiment_id=406 camera_name=random &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 agent=drqv2 seed=11 experiment_id=406 camera_name=random &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 agent=drqv2 seed=12 experiment_id=406 camera_name=random &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=drqv2 seed=10 experiment_id=407 camera_name=random &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=drqv2 seed=11 experiment_id=407 camera_name=random &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=drqv2 seed=12 experiment_id=407 camera_name=random &

export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=kitchen_kitchen-microwave-v0 agent=drqv2 seed=10 experiment_id=408 camera_name=random &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=kitchen_kitchen-microwave-v0 agent=drqv2 seed=11 experiment_id=408 camera_name=random &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=kitchen_kitchen-microwave-v0 agent=drqv2 seed=12 experiment_id=408 camera_name=random &

export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 agent=drqv2 seed=10 experiment_id=409 camera_name=random &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 agent=drqv2 seed=11 experiment_id=409 camera_name=random &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 agent=drqv2 seed=12 experiment_id=409 camera_name=random &

export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=kitchen_kitchen-microwave-v0 agent=drqv2 seed=10 experiment_id=404 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=kitchen_kitchen-microwave-v0 agent=drqv2 seed=11 experiment_id=404 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=kitchen_kitchen-microwave-v0 agent=drqv2 seed=12 experiment_id=404 camera_name=fixed &

wait
