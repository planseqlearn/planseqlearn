#! /bin/bash
#SBATCH --output=/checkpoint/sbahl/ssrl/slurm_logs/%x.out
#SBATCH --error=/checkpoint/sbahl/ssrl/slurm_logs/%x.err
#SBATCH --partition=learnfair
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=10
#SBATCH --signal=B:USR1@60
#SBATCH --open-mode=append
#SBATCH --time=3960
#SBATCH --mem=100G
#SBATCH --comment=""


# Debug output
echo $SLURM_JOB_ID $SLURM_JOB_NAME $SLURMD_NODENAME $CUDA_VISIBLE_DEVICES
echo ${args}

# Load modules
source /etc/profile.d/modules.sh
source /private/home/sbahl/.bashrc
source /etc/profile

# setup for modules and environments
source deactivate
module purge
module load cuda/11.1
module load cudnn/v8.1.1.33-cuda.11.0
module load anaconda3
source activate drqv2

# # path setup
# export PYTHONPATH=$PYTHONPATH:/private/home/sbahl/projects/dmp_rl/rlkit
# export PYTHONPATH=$PYTHONPATH:/private/home/sbahl/projects/dmp_rl/robosuite
# export PYTHONPATH=$PYTHONPATH:/private/home/sbahl/projects/dmp_rl/rllab
# export PYTHONPATH=$PYTHONPATH:/private/home/sbahl/projects/dmp_rl/pytorch-a2c-ppo-acktr-gail
# export PYTHONPATH=$PYTHONPATH:/private/home/sbahl/projects/dmp_rl/baselines
# export PYTHONPATH=$PYTHONPATH:/private/home/sbahl/projects/dmp_rl/dyn_e
# export PYTHONPATH=$PYTHONPATH:/private/home/sbahl/projects/dmp_rl/dnc
# export PYTHONPATH=$PYTHONPATH:/private/home/sbahl/projects/dmp_rl/imednet
# export PYTHONPATH=$PYTHONPATH:/private/home/sbahl/projects/dmp_rl/metaworld
# export PYTHONPATH=$PYTHONPATH:/private/home/sbahl/projects/dmp_rl/multiworld



# mujoco setup
export MUJOCO_PY_MJKEY_PATH=/private/home/sbahl/.mujoco/mjkey.txt
export MUJOCO_PY_MJPRO_PATH=/private/home/sbahl/.mujoco/mjpro200
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/private/home/sbahl/.mujoco/mjpro200/bin


cd /private/home/sbahl2/research/DisentangledRep4RL



source activate drqv2
export MKL_THREADING_LAYER=GNU;

python disrep4rl/train.py seed=10 ${args} &
python disrep4rl/train.py seed=11 ${args} &
python disrep4rl/train.py seed=12 ${args} &
wait $!