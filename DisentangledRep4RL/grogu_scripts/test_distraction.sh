#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=45
#SBATCH --time=48:00:00
#SBATCH --mem=270G
#SBATCH --gres=gpu:3
#SBATCH --partition=abhinavlong
#SBATCH --nodelist=grogu-1-34
#SBATCH --error=/grogu/user/sbahl2/slurm_logs/test_distraction.err
#SBATCH --output=/grogu/user/sbahl2/slurm_logs/test_distraction.out

cd /home/sbahl2/research/DisentangledRep4RL

echo ${args}

source activate drqv2
export MKL_THREADING_LAYER=GNU ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export bg_dataset_path=/home/sbahl2/research/DisentangledRep4RL/DAVIS/JPEGImages/480p/

# drqv2 agent with background distractions
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cup_catch agent=drqv2 seed=10 experiment_id=testing_9000 distraction.types=\[background\] distraction.dataset_path=${bg_dataset_path} save_video=true &
# drqv2 agent with color distractions
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cartpole_swingup agent=drqv2 seed=10 experiment_id=testing_9021 distraction.types=\[color\] distraction.dataset_path=${bg_dataset_path} save_video=true &
# drqv2 agent with camera distractions
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cheetah_run agent=drqv2 seed=10 experiment_id=testing_9042 distraction.types=\[camera\] distraction.dataset_path=${bg_dataset_path} save_video=true &
# V1 agent with background distractions
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=distracting_finger_spin agent=V1 seed=10 experiment_id=testing_9063 distraction.types=\[background\] distraction.dataset_path=${bg_dataset_path} save_video=true &
# V1 agent with color distractions
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=distracting_reacher_easy agent=V1 seed=10 experiment_id=testing_9084 distraction.types=\[color\] distraction.dataset_path=${bg_dataset_path} save_video=true &
# V1 agent with camera distractions
export CUDA_VISIBLE_DEVICES=1; python disrep4rl/train.py task=distracting_walker_walk agent=V1 seed=10 experiment_id=testing_9105 distraction.types=\[camera\] distraction.dataset_path=${bg_dataset_path} save_video=true &
# drqv2AE agent with camera distractions
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=distracting_cheetah_run agent=drqv2AE seed=10 experiment_id=testing_9150 distraction.types=\[camera\] distraction.dataset_path=${bg_dataset_path} save_video=true &
# V1_random_mask agent with background distractions
export CUDA_VISIBLE_DEVICES=2; python disrep4rl/train.py task=distracting_cup_catch agent=V1_random_mask seed=10 experiment_id=testing_9162 distraction.types=\[background\] distraction.dataset_path=${bg_dataset_path} save_video=true &

wait