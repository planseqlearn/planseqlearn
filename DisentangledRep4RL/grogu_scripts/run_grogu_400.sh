
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=56
#SBATCH --time=48:00:00
#SBATCH --mem=448G
#SBATCH --gres=gpu:4
#SBATCH --partition=abhinavlong
#SBATCH --nodelist=grogu-1-24
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/1_24.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/1_24.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000


export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=13 experiment_id=360 latent_dim=512 agent.reconstruction_loss_coeff=2 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=14 experiment_id=360 latent_dim=512 agent.reconstruction_loss_coeff=2 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=15 experiment_id=360 latent_dim=512 agent.reconstruction_loss_coeff=2 &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=13 experiment_id=361 latent_dim=4096 agent.reconstruction_loss_coeff=1 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=14 experiment_id=361 latent_dim=4096 agent.reconstruction_loss_coeff=1 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=15 experiment_id=361 latent_dim=4096 agent.reconstruction_loss_coeff=1 &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=13 experiment_id=362 latent_dim=4096 agent.reconstruction_loss_coeff=3 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=14 experiment_id=362 latent_dim=4096 agent.reconstruction_loss_coeff=3 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=15 experiment_id=362 latent_dim=4096 agent.reconstruction_loss_coeff=3 &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=13 experiment_id=363 latent_dim=4096 agent.reconstruction_loss_coeff=4 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=14 experiment_id=363 latent_dim=4096 agent.reconstruction_loss_coeff=4 &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_hammer-human-v1 agent=drqv2AE seed=15 experiment_id=363 latent_dim=4096 agent.reconstruction_loss_coeff=4 &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=adroit_pen-human-v1 agent=V1 seed=10 experiment_id=364 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=adroit_pen-human-v1 agent=V1 seed=11 experiment_id=364 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=adroit_pen-human-v1 agent=V1 seed=12 experiment_id=364 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=adroit_pen-human-v1 agent=V1 seed=10 experiment_id=365 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=adroit_pen-human-v1 agent=V1 seed=11 experiment_id=365 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=adroit_pen-human-v1 agent=V1 seed=12 experiment_id=365 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=adroit_pen-human-v1 agent=V1 seed=10 experiment_id=366 agent.mask_loss_coeff=2.5e-4 agent.reconstruction_loss_coeff=1e-3 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=adroit_pen-human-v1 agent=V1 seed=11 experiment_id=366 agent.mask_loss_coeff=2.5e-4 agent.reconstruction_loss_coeff=1e-3 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=adroit_pen-human-v1 agent=V1 seed=12 experiment_id=366 agent.mask_loss_coeff=2.5e-4 agent.reconstruction_loss_coeff=1e-3 &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=adroit_pen-human-v1 agent=V1 seed=10 experiment_id=367 latent_dim=1024 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=adroit_pen-human-v1 agent=V1 seed=11 experiment_id=367 latent_dim=1024 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=adroit_pen-human-v1 agent=V1 seed=12 experiment_id=367 latent_dim=1024 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-complete-v0 agent=V1 seed=13 experiment_id=368 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 camera_name=random &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-complete-v0 agent=V1 seed=14 experiment_id=368 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 camera_name=random &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-complete-v0 agent=V1 seed=15 experiment_id=368 agent.mask_loss_coeff=2.5e-2 agent.reconstruction_loss_coeff=1e-1 camera_name=random &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-complete-v0 agent=V1 seed=10 experiment_id=369 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 camera_name=random &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-complete-v0 agent=V1 seed=11 experiment_id=369 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 camera_name=random &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-complete-v0 agent=V1 seed=12 experiment_id=369 agent.mask_loss_coeff=2.5e-3 agent.reconstruction_loss_coeff=1e-2 camera_name=random &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-complete-v0 agent=V1 seed=10 experiment_id=370 agent.mask_loss_coeff=2.5e-4 agent.reconstruction_loss_coeff=1e-3 camera_name=random &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-complete-v0 agent=V1 seed=11 experiment_id=370 agent.mask_loss_coeff=2.5e-4 agent.reconstruction_loss_coeff=1e-3 camera_name=random &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-complete-v0 agent=V1 seed=12 experiment_id=370 agent.mask_loss_coeff=2.5e-4 agent.reconstruction_loss_coeff=1e-3 camera_name=random &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-complete-v0 agent=V1 seed=10 experiment_id=371 agent.mask_loss_coeff=2e-1 agent.reconstruction_loss_coeff=2 camera_name=random &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-complete-v0 agent=V1 seed=11 experiment_id=371 agent.mask_loss_coeff=2e-1 agent.reconstruction_loss_coeff=2 camera_name=random &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-complete-v0 agent=V1 seed=12 experiment_id=371 agent.mask_loss_coeff=2e-1 agent.reconstruction_loss_coeff=2 camera_name=random &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_mt3-customized agent=drqv2AE seed=10 experiment_id=372 latent_dim=512 agent.reconstruction_loss_coeff=0.02 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_mt3-customized agent=drqv2AE seed=11 experiment_id=372 latent_dim=512 agent.reconstruction_loss_coeff=0.02 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_mt3-customized agent=drqv2AE seed=12 experiment_id=372 latent_dim=512 agent.reconstruction_loss_coeff=0.02 &

export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_mt3-customized agent=drqv2AE seed=10 experiment_id=373 latent_dim=512 agent.reconstruction_loss_coeff=0.2 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_mt3-customized agent=drqv2AE seed=11 experiment_id=373 latent_dim=512 agent.reconstruction_loss_coeff=0.2 &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_mt3-customized agent=drqv2AE seed=12 experiment_id=373 latent_dim=512 agent.reconstruction_loss_coeff=0.2 &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_mt10 agent=drqv2AE seed=10 experiment_id=374 latent_dim=512 agent.reconstruction_loss_coeff=0.2&
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_mt10 agent=drqv2AE seed=11 experiment_id=374 latent_dim=512 agent.reconstruction_loss_coeff=0.2&
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_mt10 agent=drqv2AE seed=12 experiment_id=374 latent_dim=512 agent.reconstruction_loss_coeff=0.2&


wait



