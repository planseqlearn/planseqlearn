#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --time=48:00:00
#SBATCH --mem=948G
#SBATCH --gres=gpu:8
#SBATCH --partition=abhinavlong
#SBATCH --nodelist=grogu-2-6
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/2_6.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/2_6.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000


export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=10 experiment_id=345 latent_dim=512 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=11 experiment_id=345 latent_dim=512 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=12 experiment_id=345 latent_dim=512 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=10 experiment_id=346 latent_dim=1024 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=11 experiment_id=346 latent_dim=1024 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=12 experiment_id=346 latent_dim=1024 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &

export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=10 experiment_id=347 latent_dim=2048 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=11 experiment_id=347 latent_dim=2048 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=12 experiment_id=347 latent_dim=2048 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &

export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_button-press-wall-v2 agent=V1 seed=10 experiment_id=348 latent_dim=4096 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_button-press-wall-v2 agent=V1 seed=11 experiment_id=348 latent_dim=4096 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_button-press-wall-v2 agent=V1 seed=12 experiment_id=348 latent_dim=4096 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &

export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_button-press-wall-v2 agent=V1 seed=10 experiment_id=349 latent_dim=4096 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_button-press-wall-v2 agent=V1 seed=11 experiment_id=349 latent_dim=4096 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_button-press-wall-v2 agent=V1 seed=12 experiment_id=349 latent_dim=4096 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 &

export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_button-press-wall-v2 agent=V1 seed=10 experiment_id=350 latent_dim=4096 agent.mask_loss_coeff=2.5e-4 agent.reconstruction_loss_coeff=1e-3 &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_button-press-wall-v2 agent=V1 seed=11 experiment_id=350 latent_dim=4096 agent.mask_loss_coeff=2.5e-4 agent.reconstruction_loss_coeff=1e-3 &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_button-press-wall-v2 agent=V1 seed=12 experiment_id=350 latent_dim=4096 agent.mask_loss_coeff=2.5e-4 agent.reconstruction_loss_coeff=1e-3 &

export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_button-press-wall-v2 agent=V1 seed=10 experiment_id=351 latent_dim=4096 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 &
export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_button-press-wall-v2 agent=V1 seed=11 experiment_id=351 latent_dim=4096 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 &
export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_button-press-wall-v2 agent=V1 seed=12 experiment_id=351 latent_dim=4096 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 &

export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=metaworld_button-press-wall-v2 agent=V1 seed=10 experiment_id=352 latent_dim=256 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &
export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=metaworld_button-press-wall-v2 agent=V1 seed=11 experiment_id=352 latent_dim=256 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &
export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=metaworld_button-press-wall-v2 agent=V1 seed=12 experiment_id=352 latent_dim=256 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &

wait