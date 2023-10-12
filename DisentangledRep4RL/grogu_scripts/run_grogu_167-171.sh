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

export pretrained_path=/home/sbahl2/research/DisentangledRep4RL/pretrained_encoders/158_11_encoder.pt

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=10 experiment_id=167 pretrained_encoder_path=${pretrained_path} &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=11 experiment_id=167 pretrained_encoder_path=${pretrained_path} &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=12 experiment_id=167 pretrained_encoder_path=${pretrained_path} &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=10 experiment_id=168 pretrained_encoder_path=${pretrained_path} agent.detach_decoders=true &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=11 experiment_id=168 pretrained_encoder_path=${pretrained_path} agent.detach_decoders=true &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=12 experiment_id=168 pretrained_encoder_path=${pretrained_path} agent.detach_decoders=true &

export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=10 experiment_id=169 pretrained_encoder_path=${pretrained_path} agent.detach_decoders=true agent.detach_critic=true &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=11 experiment_id=169 pretrained_encoder_path=${pretrained_path} agent.detach_decoders=true agent.detach_critic=true &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=12 experiment_id=169 pretrained_encoder_path=${pretrained_path} agent.detach_decoders=true agent.detach_critic=true &

export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=10 experiment_id=170 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=11 experiment_id=170 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=12 experiment_id=170 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 &

export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=10 experiment_id=171 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 &
export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=11 experiment_id=171 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 &
export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=metaworld_mt10 agent=V1 seed=12 experiment_id=171 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 &

wait
