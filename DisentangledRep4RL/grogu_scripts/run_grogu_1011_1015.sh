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

export CUDA_VISIBLE_DEVICES=0; python train.py --domain_name metaworld --task_name button-press-topdown-wall-v2 --seed 10 --experiment_id 10110 --camera_name corner &
export CUDA_VISIBLE_DEVICES=0; python train.py --domain_name metaworld --task_name button-press-topdown-wall-v2 --seed 11 --experiment_id 10110 --camera_name corner &
export CUDA_VISIBLE_DEVICES=1; python train.py --domain_name metaworld --task_name button-press-topdown-wall-v2 --seed 12 --experiment_id 10110 --camera_name corner &

export CUDA_VISIBLE_DEVICES=1; python train.py --domain_name metaworld --task_name button-press-wall-v2 --seed 10 --experiment_id 10120 --camera_name corner &
export CUDA_VISIBLE_DEVICES=2; python train.py --domain_name metaworld --task_name button-press-wall-v2 --seed 11 --experiment_id 10120 --camera_name corner &
export CUDA_VISIBLE_DEVICES=2; python train.py --domain_name metaworld --task_name button-press-wall-v2 --seed 12 --experiment_id 10120 --camera_name corner &

export CUDA_VISIBLE_DEVICES=3; python train.py --domain_name metaworld --task_name door-close-v2 --seed 10 --experiment_id 10130 --camera_name corner &
export CUDA_VISIBLE_DEVICES=3; python train.py --domain_name metaworld --task_name door-close-v2 --seed 11 --experiment_id 10130 --camera_name corner &
export CUDA_VISIBLE_DEVICES=4; python train.py --domain_name metaworld --task_name door-close-v2 --seed 12 --experiment_id 10130 --camera_name corner &

export CUDA_VISIBLE_DEVICES=4; python train.py --domain_name metaworld --task_name door-lock-v2 --seed 10 --experiment_id 10140 --camera_name corner &
export CUDA_VISIBLE_DEVICES=5; python train.py --domain_name metaworld --task_name door-lock-v2 --seed 11 --experiment_id 10140 --camera_name corner &
export CUDA_VISIBLE_DEVICES=5; python train.py --domain_name metaworld --task_name door-lock-v2 --seed 12 --experiment_id 10140 --camera_name corner &

export CUDA_VISIBLE_DEVICES=6; python train.py --domain_name metaworld --task_name door-open-v2 --seed 10 --experiment_id 10150 --camera_name corner &
export CUDA_VISIBLE_DEVICES=6; python train.py --domain_name metaworld --task_name door-open-v2 --seed 11 --experiment_id 10150 --camera_name corner &
export CUDA_VISIBLE_DEVICES=7; python train.py --domain_name metaworld --task_name door-open-v2 --seed 12 --experiment_id 10150 --camera_name corner &


wait