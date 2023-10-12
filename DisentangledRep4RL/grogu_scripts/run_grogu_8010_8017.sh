#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --time=48:00:00
#SBATCH --mem=948G
#SBATCH --gres=gpu:8
#SBATCH --partition=deepaklong
#SBATCH --nodelist=grogu-1-40
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/1_40.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/1_40.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_box-close-v2 seed=10 experiment_id=8010 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_box-close-v2 seed=11 experiment_id=8010 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_box-close-v2 seed=12 experiment_id=8010 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_button-press-topdown-wall-v2 seed=10 experiment_id=8011 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_button-press-topdown-wall-v2 seed=11 experiment_id=8011 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_button-press-topdown-wall-v2 seed=12 experiment_id=8011 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_button-press-wall-v2 seed=10 experiment_id=8012 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_button-press-wall-v2 seed=11 experiment_id=8012 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_button-press-wall-v2 seed=12 experiment_id=8012 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_door-close-v2 seed=10 experiment_id=8013 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_door-close-v2 seed=11 experiment_id=8013 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_door-close-v2 seed=12 experiment_id=8013 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_door-lock-v2 seed=10 experiment_id=8014 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_door-lock-v2 seed=11 experiment_id=8014 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_door-lock-v2 seed=12 experiment_id=8014 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_door-open-v2 seed=10 experiment_id=8015 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_door-open-v2 seed=11 experiment_id=8015 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_door-open-v2 seed=12 experiment_id=8015 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_door-unlock-v2 seed=10 experiment_id=8016 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_door-unlock-v2 seed=11 experiment_id=8016 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_door-unlock-v2 seed=12 experiment_id=8016 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=metaworld_hammer-v2 seed=10 experiment_id=8017 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=metaworld_hammer-v2 seed=11 experiment_id=8017 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=metaworld_hammer-v2 seed=12 experiment_id=8017 camera_name=corner agent=V1_random_mask &

wait