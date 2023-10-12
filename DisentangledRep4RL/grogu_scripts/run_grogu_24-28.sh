#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --time=48:00:00
#SBATCH --mem=448G
#SBATCH --gres=gpu:8
#SBATCH --partition=abhinavlong
#SBATCH --nodelist=grogu-0-14
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/0_14.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/0_14.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export CUDA_VISIBLE_DEVICES=0; python train.py task=metaworld_bin-picking-v2 disentangled_version=V1 latent_dim=4096 seed=10 experiment_id=24 &
export CUDA_VISIBLE_DEVICES=0; python train.py task=metaworld_bin-picking-v2 disentangled_version=V1 latent_dim=4096 seed=11 experiment_id=24 &
export CUDA_VISIBLE_DEVICES=1; python train.py task=metaworld_bin-picking-v2 disentangled_version=V1 latent_dim=4096 seed=12 experiment_id=24 &
export CUDA_VISIBLE_DEVICES=1; python train.py task=metaworld_box-close-v2 disentangled_version=original_drqv2 seed=10 experiment_id=25 &
export CUDA_VISIBLE_DEVICES=2; python train.py task=metaworld_box-close-v2 disentangled_version=original_drqv2 seed=11 experiment_id=25 &
export CUDA_VISIBLE_DEVICES=2; python train.py task=metaworld_box-close-v2 disentangled_version=original_drqv2 seed=12 experiment_id=25 &
export CUDA_VISIBLE_DEVICES=3; python train.py task=metaworld_box-close-v2 disentangled_version=V1 latent_dim=4096 seed=10 experiment_id=26 &
export CUDA_VISIBLE_DEVICES=3; python train.py task=metaworld_box-close-v2 disentangled_version=V1 latent_dim=4096 seed=11 experiment_id=26 &
export CUDA_VISIBLE_DEVICES=4; python train.py task=metaworld_box-close-v2 disentangled_version=V1 latent_dim=4096 seed=12 experiment_id=26 &
export CUDA_VISIBLE_DEVICES=4; python train.py task=metaworld_button-press-topdown-v2 disentangled_version=original_drqv2 seed=10 experiment_id=27 &
export CUDA_VISIBLE_DEVICES=5; python train.py task=metaworld_button-press-topdown-v2 disentangled_version=original_drqv2 seed=11 experiment_id=27 &
export CUDA_VISIBLE_DEVICES=5; python train.py task=metaworld_button-press-topdown-v2 disentangled_version=original_drqv2 seed=12 experiment_id=27 &
export CUDA_VISIBLE_DEVICES=6; python train.py task=metaworld_button-press-topdown-v2 disentangled_version=V1 latent_dim=4096 seed=10 experiment_id=28 &
export CUDA_VISIBLE_DEVICES=6; python train.py task=metaworld_button-press-topdown-v2 disentangled_version=V1 latent_dim=4096 seed=11 experiment_id=28 &
export CUDA_VISIBLE_DEVICES=7; python train.py task=metaworld_button-press-topdown-v2 disentangled_version=V1 latent_dim=4096 seed=12 experiment_id=28 &

# export CUDA_VISIBLE_DEVICES=0; python train.py task=metaworld_handle-press-v2 disentangled_version=V1 seed=10 experiment_id=43 &
# export CUDA_VISIBLE_DEVICES=0; python train.py task=metaworld_handle-pull-side-v2 disentangled_version=V1 seed=10 experiment_id=44 &
# export CUDA_VISIBLE_DEVICES=1; python train.py task=metaworld_handle-pull-v2 disentangled_version=V1 seed=10 experiment_id=45 &
# export CUDA_VISIBLE_DEVICES=1; python train.py task=metaworld_lever-pull-v2 disentangled_version=V1 seed=10 experiment_id=46 &
# export CUDA_VISIBLE_DEVICES=2; python train.py task=metaworld_peg-insert-side-v2 disentangled_version=V1 seed=10 experiment_id=47 &
# export CUDA_VISIBLE_DEVICES=2; python train.py task=metaworld_pick-place-wall-v2 disentangled_version=V1 seed=10 experiment_id=48 &
# export CUDA_VISIBLE_DEVICES=3; python train.py task=metaworld_pick-out-of-hole-v2 disentangled_version=V1 seed=10 experiment_id=49 &
# export CUDA_VISIBLE_DEVICES=3; python train.py task=metaworld_reach-v2 disentangled_version=V1 seed=10 experiment_id=50 &
# export CUDA_VISIBLE_DEVICES=4; python train.py task=metaworld_push-back-v2 disentangled_version=V1 seed=10 experiment_id=51 &
# export CUDA_VISIBLE_DEVICES=4; python train.py task=metaworld_push-v2 disentangled_version=V1 seed=10 experiment_id=52 &
# export CUDA_VISIBLE_DEVICES=5; python train.py task=metaworld_pick-place-v2 disentangled_version=V1 seed=10 experiment_id=53 &
# export CUDA_VISIBLE_DEVICES=5; python train.py task=metaworld_plate-slide-v2 disentangled_version=V1 seed=10 experiment_id=54 &
# export CUDA_VISIBLE_DEVICES=6; python train.py task=metaworld_plate-slide-side-v2 disentangled_version=V1 seed=10 experiment_id=55 &
# export CUDA_VISIBLE_DEVICES=6; python train.py task=metaworld_plate-slide-back-v2 disentangled_version=V1 seed=10 experiment_id=56 &
# export CUDA_VISIBLE_DEVICES=7; python train.py task=metaworld_plate-slide-back-side-v2 disentangled_version=V1 seed=10 experiment_id=57 &

wait
