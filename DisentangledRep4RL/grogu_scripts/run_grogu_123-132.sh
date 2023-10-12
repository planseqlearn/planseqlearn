#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=60
#SBATCH --time=48:00:00
#SBATCH --mem=350G
#SBATCH --gres=gpu:4
#SBATCH --partition=deepaklong
#SBATCH --nodelist=grogu-1-40
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/1_40.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/1_40.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_reach-v2 agent=drqv2 seed=10 experiment_id=123 save_snapshot=true &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_push-v2 agent=drqv2 seed=10 experiment_id=124 save_snapshot=true &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_pick-place-v2 agent=drqv2 seed=10 experiment_id=125 save_snapshot=true &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_door-open-v2 agent=drqv2 seed=10 experiment_id=126 save_snapshot=true &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_drawer-open-v2 agent=drqv2 seed=10 experiment_id=127 save_snapshot=true &
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=metaworld_drawer-close-v2 agent=drqv2 seed=10 experiment_id=128 save_snapshot=true &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_button-press-topdown-v2 agent=drqv2 seed=10 experiment_id=129 save_snapshot=true &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_peg-insert-side-v2 agent=drqv2 seed=10 experiment_id=130 save_snapshot=true &
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=metaworld_window-open-v2 agent=drqv2 seed=10 experiment_id=131 save_snapshot=true &
export CUDA_VISIBLE_DEVICES=3; python disrep4rl/train.py task=metaworld_window-close-v2 agent=drqv2 seed=10 experiment_id=132 save_snapshot=true &
wait
