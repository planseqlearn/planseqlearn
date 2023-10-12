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


export pretrained_path_ae=/home/sbahl2/research/DisentangledRep4RL/exp_local/374_11_2023.01.17_19:37:09/latest.pt
export pretrained_path_dq=/home/sbahl2/research/DisentangledRep4RL/exp_local/119_13_2023.01.04_16:50:23/snapshot.pt
export pretrained_path_v1=/home/sbahl2/research/DisentangledRep4RL/exp_local/188_10_2023.01.12_22:16:57/snapshot.pt


export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_disassemble-v2 agent=V1 seed=10 experiment_id=488 pretrain.path=${pretrained_path_v1} latent_dim=512 latent_dim=512 agent.mask_loss_coeff=2.5e-4 agent.reconstruction_loss_coeff=1e-3  &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_disassemble-v2 agent=V1 seed=11 experiment_id=488 pretrain.path=${pretrained_path_v1} latent_dim=512 latent_dim=512 agent.mask_loss_coeff=2.5e-4 agent.reconstruction_loss_coeff=1e-3  &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_disassemble-v2 agent=V1 seed=12 experiment_id=488 pretrain.path=${pretrained_path_v1} latent_dim=512 latent_dim=512 agent.mask_loss_coeff=2.5e-4 agent.reconstruction_loss_coeff=1e-3  &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_disassemble-v2 agent=V1 seed=10 experiment_id=489 pretrain.path=${pretrained_path_v1} latent_dim=512 latent_dim=512 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2  &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_disassemble-v2 agent=V1 seed=11 experiment_id=489 pretrain.path=${pretrained_path_v1} latent_dim=512 latent_dim=512 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2  &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_disassemble-v2 agent=V1 seed=12 experiment_id=489 pretrain.path=${pretrained_path_v1} latent_dim=512 latent_dim=512 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2  &


cd /home/sbahl2/research/curl; export CUDA_VISIBLE_DEVICES=3; python train.py --domain_name metaworld --task_name bin-picking-v2 --seed 10 --experiment_id 1000 --camera_name corner &
cd /home/sbahl2/research/curl; export CUDA_VISIBLE_DEVICES=3; python train.py --domain_name metaworld --task_name bin-picking-v2 --seed 11 --experiment_id 1000 --camera_name corner &
cd /home/sbahl2/research/curl; export CUDA_VISIBLE_DEVICES=4; python train.py --domain_name metaworld --task_name bin-picking-v2 --seed 12 --experiment_id 1000 --camera_name corner &

cd /home/sbahl2/research/curl; export CUDA_VISIBLE_DEVICES=4; python train.py --domain_name kitchen --task_name kitchen-light-v0 --seed 10 --experiment_id 1001 --camera_name random &
cd /home/sbahl2/research/curl; export CUDA_VISIBLE_DEVICES=5; python train.py --domain_name kitchen --task_name kitchen-light-v0 --seed 11 --experiment_id 1001 --camera_name random &
cd /home/sbahl2/research/curl; export CUDA_VISIBLE_DEVICES=5; python train.py --domain_name kitchen --task_name kitchen-light-v0 --seed 12 --experiment_id 1001 --camera_name random &

cd /home/sbahl2/research/curl; export CUDA_VISIBLE_DEVICES=6; python train.py --domain_name adroit --task_name pen-human-v1 --seed 10 --experiment_id 1002 --camera_name fixed &
cd /home/sbahl2/research/curl; export CUDA_VISIBLE_DEVICES=6; python train.py --domain_name adroit --task_name pen-human-v1 --seed 11 --experiment_id 1002 --camera_name fixed &
cd /home/sbahl2/research/curl; export CUDA_VISIBLE_DEVICES=7; python train.py --domain_name adroit --task_name pen-human-v1 --seed 12 --experiment_id 1002 --camera_name fixed &

wait
