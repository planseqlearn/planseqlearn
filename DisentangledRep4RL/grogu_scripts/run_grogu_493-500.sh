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

export pretrained_path_ae=/home/sbahl2/research/DisentangledRep4RL/exp_local/374_11_2023.01.17_19:37:09/latest.pt
export pretrained_path_dq=/home/sbahl2/research/DisentangledRep4RL/exp_local/119_13_2023.01.04_16:50:23/snapshot.pt
export pretrained_path_v1=/home/sbahl2/research/DisentangledRep4RL/exp_local/188_10_2023.01.12_22:16:57/snapshot.pt


export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_disassemble-v2 agent=V1 seed=10 experiment_id=493 pretrain.path=${pretrained_path_v1} latent_dim=512 latent_dim=512 agent.mask_loss_coeff=5e-4 agent.reconstruction_loss_coeff=2e-3  &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_disassemble-v2 agent=V1 seed=11 experiment_id=493 pretrain.path=${pretrained_path_v1} latent_dim=512 latent_dim=512 agent.mask_loss_coeff=5e-4 agent.reconstruction_loss_coeff=2e-3  &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_disassemble-v2 agent=V1 seed=12 experiment_id=493 pretrain.path=${pretrained_path_v1} latent_dim=512 latent_dim=512 agent.mask_loss_coeff=5e-4 agent.reconstruction_loss_coeff=2e-3  &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_disassemble-v2 agent=V1 seed=10 experiment_id=494 pretrain.path=${pretrained_path_v1} latent_dim=512 latent_dim=512 agent.mask_loss_coeff=3.5e-3 agent.reconstruction_loss_coeff=2e-2  &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_disassemble-v2 agent=V1 seed=11 experiment_id=494 pretrain.path=${pretrained_path_v1} latent_dim=512 latent_dim=512 agent.mask_loss_coeff=3.5e-3 agent.reconstruction_loss_coeff=2e-2  &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_disassemble-v2 agent=V1 seed=12 experiment_id=494 pretrain.path=${pretrained_path_v1} latent_dim=512 latent_dim=512 agent.mask_loss_coeff=3.5e-3 agent.reconstruction_loss_coeff=2e-2  &

export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_soccer-v2 agent=V1 seed=10 experiment_id=495 pretrain.path=${pretrained_path_v1} latent_dim=512 agent.mask_loss_coeff=5e-4 agent.reconstruction_loss_coeff=2e-3  &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_soccer-v2 agent=V1 seed=11 experiment_id=495 pretrain.path=${pretrained_path_v1} latent_dim=512 agent.mask_loss_coeff=5e-4 agent.reconstruction_loss_coeff=2e-3  &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_soccer-v2 agent=V1 seed=12 experiment_id=495 pretrain.path=${pretrained_path_v1} latent_dim=512 agent.mask_loss_coeff=5e-4 agent.reconstruction_loss_coeff=2e-3  &

export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_soccer-v2 agent=V1 seed=10 experiment_id=496 pretrain.path=${pretrained_path_v1} latent_dim=512 agent.mask_loss_coeff=3.5e-3 agent.reconstruction_loss_coeff=2e-2 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_soccer-v2 agent=V1 seed=11 experiment_id=496 pretrain.path=${pretrained_path_v1} latent_dim=512 agent.mask_loss_coeff=3.5e-3 agent.reconstruction_loss_coeff=2e-2 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_soccer-v2 agent=V1 seed=12 experiment_id=496 pretrain.path=${pretrained_path_v1} latent_dim=512 agent.mask_loss_coeff=3.5e-3 agent.reconstruction_loss_coeff=2e-2 &

export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_basketball-v2 agent=V1 seed=10 experiment_id=497 pretrain.path=${pretrained_path_v1} latent_dim=512 latent_dim=512 agent.mask_loss_coeff=5e-4 agent.reconstruction_loss_coeff=2e-3  &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_basketball-v2 agent=V1 seed=11 experiment_id=497 pretrain.path=${pretrained_path_v1} latent_dim=512 latent_dim=512 agent.mask_loss_coeff=5e-4 agent.reconstruction_loss_coeff=2e-3  &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_basketball-v2 agent=V1 seed=12 experiment_id=497 pretrain.path=${pretrained_path_v1} latent_dim=512 latent_dim=512 agent.mask_loss_coeff=5e-4 agent.reconstruction_loss_coeff=2e-3  &

export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_basketball-v2 agent=V1 seed=10 experiment_id=498 pretrain.path=${pretrained_path_v1} latent_dim=512 latent_dim=512 agent.mask_loss_coeff=3.5e-3 agent.reconstruction_loss_coeff=2e-2  &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_basketball-v2 agent=V1 seed=11 experiment_id=498 pretrain.path=${pretrained_path_v1} latent_dim=512 latent_dim=512 agent.mask_loss_coeff=3.5e-3 agent.reconstruction_loss_coeff=2e-2  &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_basketball-v2 agent=V1 seed=12 experiment_id=498 pretrain.path=${pretrained_path_v1} latent_dim=512 latent_dim=512 agent.mask_loss_coeff=3.5e-3 agent.reconstruction_loss_coeff=2e-2  &

export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=10 experiment_id=499 pretrain.path=${pretrained_path_v1} latent_dim=512 latent_dim=512 latent_dim=512 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &
export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=11 experiment_id=499 pretrain.path=${pretrained_path_v1} latent_dim=512 latent_dim=512 latent_dim=512 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &
export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=12 experiment_id=499 pretrain.path=${pretrained_path_v1} latent_dim=512 latent_dim=512 latent_dim=512 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &

export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=10 experiment_id=500 pretrain.path=${pretrained_path_v1} latent_dim=512 latent_dim=512 latent_dim=512 agent.mask_loss_coeff=2.5e-4 agent.reconstruction_loss_coeff=1e-3   &
export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=11 experiment_id=500 pretrain.path=${pretrained_path_v1} latent_dim=512 latent_dim=512 latent_dim=512 agent.mask_loss_coeff=2.5e-4 agent.reconstruction_loss_coeff=1e-3   &
export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=12 experiment_id=500 pretrain.path=${pretrained_path_v1} latent_dim=512 latent_dim=512 latent_dim=512 agent.mask_loss_coeff=2.5e-4 agent.reconstruction_loss_coeff=1e-3   &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=10 experiment_id=501 pretrain.path=${pretrained_path_v1} latent_dim=512 latent_dim=512 agent.mask_loss_coeff=5e-4 agent.reconstruction_loss_coeff=2e-3  &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=11 experiment_id=501 pretrain.path=${pretrained_path_v1} latent_dim=512 latent_dim=512 agent.mask_loss_coeff=5e-4 agent.reconstruction_loss_coeff=2e-3  &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=12 experiment_id=501 pretrain.path=${pretrained_path_v1} latent_dim=512 latent_dim=512 agent.mask_loss_coeff=5e-4 agent.reconstruction_loss_coeff=2e-3  &

export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=10 experiment_id=502 pretrain.path=${pretrained_path_v1} latent_dim=512 latent_dim=512 agent.mask_loss_coeff=3.5e-3 agent.reconstruction_loss_coeff=2e-2  &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=11 experiment_id=502 pretrain.path=${pretrained_path_v1} latent_dim=512 latent_dim=512 agent.mask_loss_coeff=3.5e-3 agent.reconstruction_loss_coeff=2e-2  &
export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=12 experiment_id=502 pretrain.path=${pretrained_path_v1} latent_dim=512 latent_dim=512 agent.mask_loss_coeff=3.5e-3 agent.reconstruction_loss_coeff=2e-2  &

wait