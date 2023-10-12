#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --time=48:00:00
#SBATCH --mem=448G
#SBATCH --gres=gpu:8
#SBATCH --partition=abhinavlong
#SBATCH --nodelist=grogu-1-24
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/1_24.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/1_24.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_assembly-v2 agent=V1 seed=10 experiment_id=295 latent_dim=4096 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_assembly-v2 agent=V1 seed=11 experiment_id=295 latent_dim=4096 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_assembly-v2 agent=V1 seed=12 experiment_id=295 latent_dim=4096 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_assembly-v2 agent=V1 seed=10 experiment_id=296 latent_dim=4096 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_assembly-v2 agent=V1 seed=11 experiment_id=296 latent_dim=4096 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_assembly-v2 agent=V1 seed=12 experiment_id=296 latent_dim=4096 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 &

export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_assembly-v2 agent=V1 seed=10 experiment_id=297 latent_dim=4096 agent.mask_loss_coeff=2.5e-4 agent.reconstruction_loss_coeff=1e-3 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_assembly-v2 agent=V1 seed=11 experiment_id=297 latent_dim=4096 agent.mask_loss_coeff=2.5e-4 agent.reconstruction_loss_coeff=1e-3 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_assembly-v2 agent=V1 seed=12 experiment_id=297 latent_dim=4096 agent.mask_loss_coeff=2.5e-4 agent.reconstruction_loss_coeff=1e-3 &

export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_assembly-v2 agent=V1 seed=10 experiment_id=298 latent_dim=4096 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_assembly-v2 agent=V1 seed=11 experiment_id=298 latent_dim=4096 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_assembly-v2 agent=V1 seed=12 experiment_id=298 latent_dim=4096 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 &

export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=10 experiment_id=299 latent_dim=4096 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=11 experiment_id=299 latent_dim=4096 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=12 experiment_id=299 latent_dim=4096 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &

export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=10 experiment_id=300 latent_dim=4096 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=11 experiment_id=300 latent_dim=4096 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=12 experiment_id=300 latent_dim=4096 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 &

export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=10 experiment_id=301 latent_dim=4096 agent.mask_loss_coeff=2.5e-4 agent.reconstruction_loss_coeff=1e-3 &
export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=11 experiment_id=301 latent_dim=4096 agent.mask_loss_coeff=2.5e-4 agent.reconstruction_loss_coeff=1e-3 &
export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=12 experiment_id=301 latent_dim=4096 agent.mask_loss_coeff=2.5e-4 agent.reconstruction_loss_coeff=1e-3 &

export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=10 experiment_id=302 latent_dim=4096 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 &
export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=11 experiment_id=302 latent_dim=4096 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 &
export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=12 experiment_id=302 latent_dim=4096 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 &

wait

# python disrep4rl/train.py task=metaworld_assembly-v2 agent=drqv2AE seed=10 experiment_id=273 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &
# python disrep4rl/train.py task=metaworld_bin-picking-v2 agent=drqv2AE seed=10 experiment_id=274 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &
# python disrep4rl/train.py task=metaworld_box-close-v2 agent=drqv2AE seed=10 experiment_id=275 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &
# python disrep4rl/train.py task=metaworld_button-press-topdown-wall-v2 agent=drqv2AE seed=10 experiment_id=276 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &
# python disrep4rl/train.py task=metaworld_button-press-wall-v2 agent=drqv2AE seed=10 experiment_id=277 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &
# python disrep4rl/train.py task=metaworld_door-close-v2 agent=drqv2AE seed=10 experiment_id=278 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &
# python disrep4rl/train.py task=metaworld_door-lock-v2 agent=drqv2AE seed=10 experiment_id=279 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &
# python disrep4rl/train.py task=metaworld_door-open-v2 agent=drqv2AE seed=10 experiment_id=280 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &
# python disrep4rl/train.py task=metaworld_door-unlock-v2 agent=drqv2AE seed=10 experiment_id=281 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &
# python disrep4rl/train.py task=metaworld_hammer-v2 agent=drqv2AE seed=10 experiment_id=282 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &