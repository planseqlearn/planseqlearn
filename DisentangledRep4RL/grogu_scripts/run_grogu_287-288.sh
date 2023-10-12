#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --mem=179G
#SBATCH --gres=gpu:6
#SBATCH --partition=deepaklong
#SBATCH --nodelist=grogu-1-19
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/1_19.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/1_19.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-complete-v0 agent=drqv2AE seed=10 experiment_id=287 agent.reconstruction_loss_coeff=2 camera_name=random &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=kitchen_kitchen-complete-v0 agent=drqv2AE seed=11 experiment_id=287 agent.reconstruction_loss_coeff=2 camera_name=random &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=kitchen_kitchen-complete-v0 agent=drqv2AE seed=12 experiment_id=287 agent.reconstruction_loss_coeff=2 camera_name=random &

export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=kitchen_kitchen-mixed-v0 agent=drqv2AE seed=10 experiment_id=288 agent.reconstruction_loss_coeff=2 camera_name=random &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=kitchen_kitchen-mixed-v0 agent=drqv2AE seed=11 experiment_id=288 agent.reconstruction_loss_coeff=2 camera_name=random &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=kitchen_kitchen-mixed-v0 agent=drqv2AE seed=12 experiment_id=288 agent.reconstruction_loss_coeff=2 camera_name=random &

wait
