#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --time=48:00:00
#SBATCH --mem=900G
#SBATCH --gres=gpu:8
#SBATCH --partition=abhinavlong
#SBATCH --nodelist=grogu-2-6
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/2_6.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/2_6.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_door-close-v2 agent=V1 latent_dim=512 seed=10 experiment_id=224 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_door-lock-v2 agent=V1 latent_dim=512 seed=10 experiment_id=225 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_door-unlock-v2 agent=V1 latent_dim=512 seed=10 experiment_id=226 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_hand-insert-v2 agent=V1 latent_dim=512 seed=10 experiment_id=227 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_faucet-open-v2 agent=V1 latent_dim=512 seed=10 experiment_id=228 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_faucet-close-v2 agent=V1 latent_dim=512 seed=10 experiment_id=229 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_hammer-v2 agent=V1 latent_dim=512 seed=10 experiment_id=230 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_handle-press-side-v2 agent=V1 latent_dim=512 seed=10 experiment_id=231 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_handle-press-v2 agent=V1 latent_dim=512 seed=10 experiment_id=232 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_handle-pull-side-v2 agent=V1 latent_dim=512 seed=10 experiment_id=233 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_handle-pull-v2 agent=V1 latent_dim=512 seed=10 experiment_id=234 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_lever-pull-v2 agent=V1 latent_dim=512 seed=10 experiment_id=235 &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_pick-place-wall-v2 agent=V1 latent_dim=512 seed=10 experiment_id=236 &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_pick-out-of-hole-v2 agent=V1 latent_dim=512 seed=10 experiment_id=237 &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_push-back-v2 agent=V1 latent_dim=512 seed=10 experiment_id=238 &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_plate-slide-v2 agent=V1 latent_dim=512 seed=10 experiment_id=239 &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_plate-slide-side-v2 agent=V1 latent_dim=512 seed=10 experiment_id=240 &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_plate-slide-back-v2 agent=V1 latent_dim=512 seed=10 experiment_id=241 &
export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_plate-slide-back-side-v2 agent=V1 latent_dim=512 seed=10 experiment_id=242 &
export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_peg-unplug-side-v2 agent=V1 latent_dim=512 seed=10 experiment_id=243 &
export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_soccer-v2 agent=V1 latent_dim=512 seed=10 experiment_id=244 &
export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=metaworld_stick-push-v2 agent=V1 latent_dim=512 seed=10 experiment_id=245 &
export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=metaworld_stick-pull-v2 agent=V1 latent_dim=512 seed=10 experiment_id=246 &
export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=metaworld_push-wall-v2 agent=V1 latent_dim=512 seed=10 experiment_id=247 &

wait
