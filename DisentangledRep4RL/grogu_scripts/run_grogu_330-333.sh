#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=30
#SBATCH --time=48:00:00
#SBATCH --mem=168G
#SBATCH --gres=gpu:3
#SBATCH --partition=deepaklong
#SBATCH --nodelist=grogu-0-19
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/0_19_2.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/0_19_2.out

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

export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=10 experiment_id=344 latent_dim=256 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &
export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=11 experiment_id=344 latent_dim=256 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &
export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=12 experiment_id=344 latent_dim=256 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &

export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_door-lock-v2 agent=drqv2AE seed=10 experiment_id=336 latent_dim=4096 agent.reconstruction_loss_coeff=0.02 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_door-lock-v2 agent=drqv2AE seed=11 experiment_id=336 latent_dim=4096 agent.reconstruction_loss_coeff=0.02 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_door-lock-v2 agent=drqv2AE seed=12 experiment_id=336 latent_dim=4096 agent.reconstruction_loss_coeff=0.02 &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_door-open-v2 agent=drqv2AE seed=10 experiment_id=337 latent_dim=4096 agent.reconstruction_loss_coeff=0.02 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_door-open-v2 agent=drqv2AE seed=11 experiment_id=337 latent_dim=4096 agent.reconstruction_loss_coeff=0.02 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_door-open-v2 agent=drqv2AE seed=12 experiment_id=337 latent_dim=4096 agent.reconstruction_loss_coeff=0.02 &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_door-unlock-v2 agent=drqv2AE seed=10 experiment_id=338 latent_dim=4096 agent.reconstruction_loss_coeff=0.02 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_door-unlock-v2 agent=drqv2AE seed=11 experiment_id=338 latent_dim=4096 agent.reconstruction_loss_coeff=0.02 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_door-unlock-v2 agent=drqv2AE seed=12 experiment_id=338 latent_dim=4096 agent.reconstruction_loss_coeff=0.02 &

export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_hammer-v2 agent=drqv2AE seed=10 experiment_id=339 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_hammer-v2 agent=drqv2AE seed=11 experiment_id=339 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_hammer-v2 agent=drqv2AE seed=12 experiment_id=339 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &


export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=10 experiment_id=340 latent_dim=4096 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=11 experiment_id=340 latent_dim=4096 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=12 experiment_id=340 latent_dim=4096 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=10 experiment_id=341 latent_dim=4096 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=11 experiment_id=341 latent_dim=4096 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=12 experiment_id=341 latent_dim=4096 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 &

export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=10 experiment_id=342 latent_dim=4096 agent.mask_loss_coeff=2.5e-4 agent.reconstruction_loss_coeff=1e-3 &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=11 experiment_id=342 latent_dim=4096 agent.mask_loss_coeff=2.5e-4 agent.reconstruction_loss_coeff=1e-3 &
export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=12 experiment_id=342 latent_dim=4096 agent.mask_loss_coeff=2.5e-4 agent.reconstruction_loss_coeff=1e-3 &

export CUDA_VISIBLE_DEVICES=4; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=10 experiment_id=343 latent_dim=4096 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=11 experiment_id=343 latent_dim=4096 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 &
export CUDA_VISIBLE_DEVICES=5; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=12 experiment_id=343 latent_dim=4096 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 &

export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=10 experiment_id=344 latent_dim=256 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &
export CUDA_VISIBLE_DEVICES=6; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=11 experiment_id=344 latent_dim=256 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &
export CUDA_VISIBLE_DEVICES=7; python disrep4rl/train.py task=metaworld_box-close-v2 agent=V1 seed=12 experiment_id=344 latent_dim=256 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &

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


export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_button-press-wall-v2 agent=V1 seed=10 experiment_id=353 latent_dim=512 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_button-press-wall-v2 agent=V1 seed=11 experiment_id=353 latent_dim=512 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_button-press-wall-v2 agent=V1 seed=12 experiment_id=353 latent_dim=512 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_button-press-wall-v2 agent=V1 seed=10 experiment_id=354 latent_dim=1024 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_button-press-wall-v2 agent=V1 seed=11 experiment_id=354 latent_dim=1024 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_button-press-wall-v2 agent=V1 seed=12 experiment_id=354 latent_dim=1024 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &

export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_button-press-wall-v2 agent=V1 seed=10 experiment_id=355 latent_dim=2048 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_button-press-wall-v2 agent=V1 seed=11 experiment_id=355 latent_dim=2048 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_button-press-wall-v2 agent=V1 seed=12 experiment_id=355 latent_dim=2048 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &


wait




# door-close: 278 
# door-lock: 279 
# door-open: 280 
# door-unlock: 281
# hammer: 282

# ours: button-press-wall, box-close



# python disrep4rl/train.py task=metaworld_button-press-topdown-wall-v2 agent=drqv2AE seed=10 experiment_id=273 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &
# python disrep4rl/train.py task=metaworld_bin-picking-v2 agent=drqv2AE seed=10 experiment_id=274 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &
# python disrep4rl/train.py task=metaworld_button-press-wall-v2 agent=drqv2AE seed=10 experiment_id=275 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &
# python disrep4rl/train.py task=metaworld_button-press-topdown-wall-v2 agent=drqv2AE seed=10 experiment_id=276 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &
# python disrep4rl/train.py task=metaworld_button-press-wall-v2 agent=drqv2AE seed=10 experiment_id=277 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &
# python disrep4rl/train.py task=metaworld_door-close-v2 agent=drqv2AE seed=10 experiment_id=278 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &
# python disrep4rl/train.py task=metaworld_door-lock-v2 agent=drqv2AE seed=10 experiment_id=279 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &
# python disrep4rl/train.py task=metaworld_door-open-v2 agent=drqv2AE seed=10 experiment_id=280 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &
# python disrep4rl/train.py task=metaworld_door-unlock-v2 agent=drqv2AE seed=10 experiment_id=281 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &
# python disrep4rl/train.py task=metaworld_hammer-v2 agent=drqv2AE seed=10 experiment_id=282 latent_dim=4096 agent.reconstruction_loss_coeff=0.002 &