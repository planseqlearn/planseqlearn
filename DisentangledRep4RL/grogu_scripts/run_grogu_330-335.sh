#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=110
#SBATCH --time=48:00:00
#SBATCH --mem=900G
#SBATCH --gres=gpu:7
#SBATCH --partition=deepaklong
#SBATCH --nodelist=grogu-1-3
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/1_3.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/1_3.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000


export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=13 experiment_id=330 latent_dim=4096 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=14 experiment_id=330 latent_dim=4096 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=15 experiment_id=330 latent_dim=4096 &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_button-press-wall-v2 agent=drqv2AE seed=10 experiment_id=326 latent_dim=1024 agent.reconstruction_loss_coeff=0.02 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_button-press-wall-v2 agent=drqv2AE seed=11 experiment_id=326 latent_dim=1024 agent.reconstruction_loss_coeff=0.02 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_button-press-wall-v2 agent=drqv2AE seed=12 experiment_id=326 latent_dim=1024 agent.reconstruction_loss_coeff=0.02 &

export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_button-press-wall-v2 agent=drqv2AE seed=10 experiment_id=327 latent_dim=512 agent.reconstruction_loss_coeff=0.02 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_button-press-wall-v2 agent=drqv2AE seed=11 experiment_id=327 latent_dim=512 agent.reconstruction_loss_coeff=0.02 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_button-press-wall-v2 agent=drqv2AE seed=12 experiment_id=327 latent_dim=512 agent.reconstruction_loss_coeff=0.02 &

export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_bin-picking-v2 agent=drqv2AE seed=10 experiment_id=332 latent_dim=4096 agent.reconstruction_loss_coeff=0.02 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_bin-picking-v2 agent=drqv2AE seed=11 experiment_id=332 latent_dim=4096 agent.reconstruction_loss_coeff=0.02 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_bin-picking-v2 agent=drqv2AE seed=12 experiment_id=332 latent_dim=4096 agent.reconstruction_loss_coeff=0.02 &

export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_button-press-topdown-wall-v2 agent=drqv2AE seed=10 experiment_id=333 latent_dim=4096 agent.reconstruction_loss_coeff=0.02 &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_button-press-topdown-wall-v2 agent=drqv2AE seed=11 experiment_id=333 latent_dim=4096 agent.reconstruction_loss_coeff=0.02 &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_button-press-topdown-wall-v2 agent=drqv2AE seed=12 experiment_id=333 latent_dim=4096 agent.reconstruction_loss_coeff=0.02 &

export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_box-close-v2 agent=drqv2AE seed=10 experiment_id=334 latent_dim=4096 agent.reconstruction_loss_coeff=0.02 &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_box-close-v2 agent=drqv2AE seed=11 experiment_id=334 latent_dim=4096 agent.reconstruction_loss_coeff=0.02 &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_box-close-v2 agent=drqv2AE seed=12 experiment_id=334 latent_dim=4096 agent.reconstruction_loss_coeff=0.02 &

export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_door-close-v2 agent=drqv2AE seed=10 experiment_id=335 latent_dim=4096 agent.reconstruction_loss_coeff=0.02 &
export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_door-close-v2 agent=drqv2AE seed=11 experiment_id=335 latent_dim=4096 agent.reconstruction_loss_coeff=0.02 &
export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_door-close-v2 agent=drqv2AE seed=12 experiment_id=335 latent_dim=4096 agent.reconstruction_loss_coeff=0.02 &



wait