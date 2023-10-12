#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --time=48:00:00
#SBATCH --mem=448G
#SBATCH --gres=gpu:8
#SBATCH --partition=deepaklong
#SBATCH --nodelist=grogu-1-19
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/1_19.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/1_19.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export pretrained_path=/home/sbahl2/research/DisentangledRep4RL/exp_local/374_11_2023.01.17_19:37:09/latest.pt

# assembly, door lock, door close, hammer, dial turn

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_door-lock-v2 agent=drqv2AE seed=10 experiment_id=448 pretrain.path=${pretrained_path} latent_dim=512 agent.reconstruction_loss_coeff=0.02&
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_door-lock-v2 agent=drqv2AE seed=11 experiment_id=448 pretrain.path=${pretrained_path} latent_dim=512 agent.reconstruction_loss_coeff=0.02&
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_door-lock-v2 agent=drqv2AE seed=12 experiment_id=448 pretrain.path=${pretrained_path} latent_dim=512 agent.reconstruction_loss_coeff=0.02&

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_bin-picking-v2 agent=drqv2AE seed=10 experiment_id=449 pretrain.path=${pretrained_path} latent_dim=512 agent.reconstruction_loss_coeff=0.02&
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_bin-picking-v2 agent=drqv2AE seed=11 experiment_id=449 pretrain.path=${pretrained_path} latent_dim=512 agent.reconstruction_loss_coeff=0.02&
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_bin-picking-v2 agent=drqv2AE seed=12 experiment_id=449 pretrain.path=${pretrained_path} latent_dim=512 agent.reconstruction_loss_coeff=0.02&

export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_box-close-v2 agent=drqv2AE seed=10 experiment_id=450 pretrain.path=${pretrained_path} latent_dim=512 agent.reconstruction_loss_coeff=0.02&
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_box-close-v2 agent=drqv2AE seed=11 experiment_id=450 pretrain.path=${pretrained_path} latent_dim=512 agent.reconstruction_loss_coeff=0.02&
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_box-close-v2 agent=drqv2AE seed=12 experiment_id=450 pretrain.path=${pretrained_path} latent_dim=512 agent.reconstruction_loss_coeff=0.02&

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_assembly-v2 agent=drqv2AE seed=10 experiment_id=451 pretrain.path=${pretrained_path} latent_dim=512 agent.reconstruction_loss_coeff=0.02&
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_assembly-v2 agent=drqv2AE seed=11 experiment_id=451 pretrain.path=${pretrained_path} latent_dim=512 agent.reconstruction_loss_coeff=0.02&
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_assembly-v2 agent=drqv2AE seed=12 experiment_id=451 pretrain.path=${pretrained_path} latent_dim=512 agent.reconstruction_loss_coeff=0.02&

export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_hammer-v2 agent=drqv2AE seed=10 experiment_id=452 pretrain.path=${pretrained_path} latent_dim=512 agent.reconstruction_loss_coeff=0.02&
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_hammer-v2 agent=drqv2AE seed=11 experiment_id=452 pretrain.path=${pretrained_path} latent_dim=512 agent.reconstruction_loss_coeff=0.02&
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_hammer-v2 agent=drqv2AE seed=12 experiment_id=452 pretrain.path=${pretrained_path} latent_dim=512 agent.reconstruction_loss_coeff=0.02&

export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_coffee-button-v2 agent=drqv2AE seed=10 experiment_id=453 pretrain.path=${pretrained_path} latent_dim=512 agent.reconstruction_loss_coeff=0.02&
export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_coffee-button-v2 agent=drqv2AE seed=11 experiment_id=453 pretrain.path=${pretrained_path} latent_dim=512 agent.reconstruction_loss_coeff=0.02&
export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=metaworld_coffee-button-v2 agent=drqv2AE seed=12 experiment_id=453 pretrain.path=${pretrained_path} latent_dim=512 agent.reconstruction_loss_coeff=0.02&

export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_peg-insert-side-v2 agent=drqv2AE seed=10 experiment_id=454 pretrain.path=${pretrained_path} latent_dim=512 agent.reconstruction_loss_coeff=0.02&
export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_peg-insert-side-v2 agent=drqv2AE seed=11 experiment_id=454 pretrain.path=${pretrained_path} latent_dim=512 agent.reconstruction_loss_coeff=0.02&
export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=metaworld_peg-insert-side-v2 agent=drqv2AE seed=12 experiment_id=454 pretrain.path=${pretrained_path} latent_dim=512 agent.reconstruction_loss_coeff=0.02&

export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_soccer-v2 agent=drqv2AE seed=10 experiment_id=455 pretrain.path=${pretrained_path} latent_dim=512 agent.reconstruction_loss_coeff=0.02&
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_soccer-v2 agent=drqv2AE seed=11 experiment_id=455 pretrain.path=${pretrained_path} latent_dim=512 agent.reconstruction_loss_coeff=0.02&
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_soccer-v2 agent=drqv2AE seed=12 experiment_id=455 pretrain.path=${pretrained_path} latent_dim=512 agent.reconstruction_loss_coeff=0.02&

export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_plate-slide-v2 agent=drqv2AE seed=10 experiment_id=456 pretrain.path=${pretrained_path} latent_dim=512 agent.reconstruction_loss_coeff=0.02&
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_plate-slide-v2 agent=drqv2AE seed=11 experiment_id=456 pretrain.path=${pretrained_path} latent_dim=512 agent.reconstruction_loss_coeff=0.02&
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_plate-slide-v2 agent=drqv2AE seed=12 experiment_id=456 pretrain.path=${pretrained_path} latent_dim=512 agent.reconstruction_loss_coeff=0.02&


wait
