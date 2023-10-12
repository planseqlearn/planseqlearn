export bg_dataset_path=/home/sbahl2/research/DisentangledRep4RL/DAVIS/JPEGImages/480p/

# drqv2 agent with background distractions

distraction.dataset_path=/home/sbahl2/research/DisentangledRep4RL/DAVIS/JPEGImages/480p/
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cup_catch agent=drqv2 seed=10 experiment_id=9000 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cup_catch agent=drqv2 seed=11 experiment_id=9000 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cup_catch agent=drqv2 seed=12 experiment_id=9000 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cartpole_swingup agent=drqv2 seed=10 experiment_id=9001 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cartpole_swingup agent=drqv2 seed=11 experiment_id=9001 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cartpole_swingup agent=drqv2 seed=12 experiment_id=9001 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cheetah_run agent=drqv2 seed=10 experiment_id=9002 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cheetah_run agent=drqv2 seed=11 experiment_id=9002 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cheetah_run agent=drqv2 seed=12 experiment_id=9002 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_finger_spin agent=drqv2 seed=10 experiment_id=9003 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_finger_spin agent=drqv2 seed=11 experiment_id=9003 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_finger_spin agent=drqv2 seed=12 experiment_id=9003 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_reacher_easy agent=drqv2 seed=10 experiment_id=9004 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_reacher_easy agent=drqv2 seed=11 experiment_id=9004 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_reacher_easy agent=drqv2 seed=12 experiment_id=9005 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_walker_walk agent=drqv2 seed=10 experiment_id=9005 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_walker_walk agent=drqv2 seed=11 experiment_id=9005 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_walker_walk agent=drqv2 seed=12 experiment_id=9005 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &


# drqv2 agent with color distractions
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cup_catch agent=drqv2 seed=10 experiment_id=9006 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cup_catch agent=drqv2 seed=11 experiment_id=9006 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cup_catch agent=drqv2 seed=12 experiment_id=9006 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cartpole_swingup agent=drqv2 seed=10 experiment_id=9007 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cartpole_swingup agent=drqv2 seed=11 experiment_id=9007 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cartpole_swingup agent=drqv2 seed=12 experiment_id=9007 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cheetah_run agent=drqv2 seed=10 experiment_id=9008 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cheetah_run agent=drqv2 seed=11 experiment_id=9008 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cheetah_run agent=drqv2 seed=12 experiment_id=9008 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_finger_spin agent=drqv2 seed=10 experiment_id=9009 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_finger_spin agent=drqv2 seed=11 experiment_id=9009 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_finger_spin agent=drqv2 seed=12 experiment_id=9009 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_reacher_easy agent=drqv2 seed=10 experiment_id=9010 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_reacher_easy agent=drqv2 seed=11 experiment_id=9010 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_reacher_easy agent=drqv2 seed=12 experiment_id=9010 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_walker_walk agent=drqv2 seed=10 experiment_id=9011 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_walker_walk agent=drqv2 seed=11 experiment_id=9011 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_walker_walk agent=drqv2 seed=12 experiment_id=9011 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &


# drqv2 agent with camera distractions
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cup_catch agent=drqv2 seed=10 experiment_id=9012 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cup_catch agent=drqv2 seed=11 experiment_id=9012 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cup_catch agent=drqv2 seed=12 experiment_id=9012 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cartpole_swingup agent=drqv2 seed=10 experiment_id=9013 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cartpole_swingup agent=drqv2 seed=11 experiment_id=9013 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cartpole_swingup agent=drqv2 seed=12 experiment_id=9013 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cheetah_run agent=drqv2 seed=10 experiment_id=9014 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cheetah_run agent=drqv2 seed=11 experiment_id=9014 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cheetah_run agent=drqv2 seed=12 experiment_id=9014 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_finger_spin agent=drqv2 seed=10 experiment_id=9015 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_finger_spin agent=drqv2 seed=11 experiment_id=9015 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_finger_spin agent=drqv2 seed=12 experiment_id=9015 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_reacher_easy agent=drqv2 seed=10 experiment_id=9016 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_reacher_easy agent=drqv2 seed=11 experiment_id=9016 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_reacher_easy agent=drqv2 seed=12 experiment_id=9016 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_walker_walk agent=drqv2 seed=10 experiment_id=9017 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_walker_walk agent=drqv2 seed=11 experiment_id=9017 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_walker_walk agent=drqv2 seed=12 experiment_id=9017 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &


# V1 agent with background distractions
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cup_catch agent=V1 seed=10 experiment_id=9018 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cup_catch agent=V1 seed=11 experiment_id=9018 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cup_catch agent=V1 seed=12 experiment_id=9018 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cartpole_swingup agent=V1 seed=10 experiment_id=9019 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cartpole_swingup agent=V1 seed=11 experiment_id=9019 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cartpole_swingup agent=V1 seed=12 experiment_id=9019 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cheetah_run agent=V1 seed=10 experiment_id=9020 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cheetah_run agent=V1 seed=11 experiment_id=9020 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cheetah_run agent=V1 seed=12 experiment_id=9020 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_finger_spin agent=V1 seed=10 experiment_id=9021 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_finger_spin agent=V1 seed=11 experiment_id=9021 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_finger_spin agent=V1 seed=12 experiment_id=9021 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_reacher_easy agent=V1 seed=10 experiment_id=9022 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_reacher_easy agent=V1 seed=11 experiment_id=9022 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_reacher_easy agent=V1 seed=12 experiment_id=9022 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_walker_walk agent=V1 seed=10 experiment_id=9023 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_walker_walk agent=V1 seed=11 experiment_id=9023 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_walker_walk agent=V1 seed=12 experiment_id=9023 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &


# V1 agent with color distractions
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cup_catch agent=V1 seed=10 experiment_id=9024 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cup_catch agent=V1 seed=11 experiment_id=9024 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cup_catch agent=V1 seed=12 experiment_id=9024 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cartpole_swingup agent=V1 seed=10 experiment_id=9025 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cartpole_swingup agent=V1 seed=11 experiment_id=9025 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cartpole_swingup agent=V1 seed=12 experiment_id=9025 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cheetah_run agent=V1 seed=10 experiment_id=9026 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cheetah_run agent=V1 seed=11 experiment_id=9026 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cheetah_run agent=V1 seed=12 experiment_id=9026 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_finger_spin agent=V1 seed=10 experiment_id=9027 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_finger_spin agent=V1 seed=11 experiment_id=9027 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_finger_spin agent=V1 seed=12 experiment_id=9027 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_reacher_easy agent=V1 seed=10 experiment_id=9028 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_reacher_easy agent=V1 seed=11 experiment_id=9028 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_reacher_easy agent=V1 seed=12 experiment_id=9028 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_walker_walk agent=V1 seed=10 experiment_id=9029 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_walker_walk agent=V1 seed=11 experiment_id=9029 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_walker_walk agent=V1 seed=12 experiment_id=9029 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &


# V1 agent with camera distractions
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cup_catch agent=V1 seed=10 experiment_id=9090 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cup_catch agent=V1 seed=11 experiment_id=9091 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cup_catch agent=V1 seed=12 experiment_id=9092 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cartpole_swingup agent=V1 seed=10 experiment_id=9093 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cartpole_swingup agent=V1 seed=11 experiment_id=9094 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cartpole_swingup agent=V1 seed=12 experiment_id=9095 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cheetah_run agent=V1 seed=10 experiment_id=9096 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cheetah_run agent=V1 seed=11 experiment_id=9097 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cheetah_run agent=V1 seed=12 experiment_id=9098 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_finger_spin agent=V1 seed=10 experiment_id=9099 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_finger_spin agent=V1 seed=11 experiment_id=9100 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_finger_spin agent=V1 seed=12 experiment_id=9101 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_reacher_easy agent=V1 seed=10 experiment_id=9102 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_reacher_easy agent=V1 seed=11 experiment_id=9103 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_reacher_easy agent=V1 seed=12 experiment_id=9104 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_walker_walk agent=V1 seed=10 experiment_id=9105 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_walker_walk agent=V1 seed=11 experiment_id=9106 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_walker_walk agent=V1 seed=12 experiment_id=9107 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &


# drqv2AE agent with background distractions
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cup_catch agent=drqv2AE seed=10 experiment_id=9108 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cup_catch agent=drqv2AE seed=11 experiment_id=9109 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cup_catch agent=drqv2AE seed=12 experiment_id=9110 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cartpole_swingup agent=drqv2AE seed=10 experiment_id=9111 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cartpole_swingup agent=drqv2AE seed=11 experiment_id=9112 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cartpole_swingup agent=drqv2AE seed=12 experiment_id=9113 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cheetah_run agent=drqv2AE seed=10 experiment_id=9114 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cheetah_run agent=drqv2AE seed=11 experiment_id=9115 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cheetah_run agent=drqv2AE seed=12 experiment_id=9116 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_finger_spin agent=drqv2AE seed=10 experiment_id=9117 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_finger_spin agent=drqv2AE seed=11 experiment_id=9118 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_finger_spin agent=drqv2AE seed=12 experiment_id=9119 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_reacher_easy agent=drqv2AE seed=10 experiment_id=9120 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_reacher_easy agent=drqv2AE seed=11 experiment_id=9121 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_reacher_easy agent=drqv2AE seed=12 experiment_id=9122 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_walker_walk agent=drqv2AE seed=10 experiment_id=9123 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_walker_walk agent=drqv2AE seed=11 experiment_id=9124 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_walker_walk agent=drqv2AE seed=12 experiment_id=9125 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &


# drqv2AE agent with color distractions
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cup_catch agent=drqv2AE seed=10 experiment_id=9126 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cup_catch agent=drqv2AE seed=11 experiment_id=9127 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cup_catch agent=drqv2AE seed=12 experiment_id=9128 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cartpole_swingup agent=drqv2AE seed=10 experiment_id=9129 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cartpole_swingup agent=drqv2AE seed=11 experiment_id=9130 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cartpole_swingup agent=drqv2AE seed=12 experiment_id=9131 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cheetah_run agent=drqv2AE seed=10 experiment_id=9132 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cheetah_run agent=drqv2AE seed=11 experiment_id=9133 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cheetah_run agent=drqv2AE seed=12 experiment_id=9134 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_finger_spin agent=drqv2AE seed=10 experiment_id=9135 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_finger_spin agent=drqv2AE seed=11 experiment_id=9136 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_finger_spin agent=drqv2AE seed=12 experiment_id=9137 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_reacher_easy agent=drqv2AE seed=10 experiment_id=9138 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_reacher_easy agent=drqv2AE seed=11 experiment_id=9139 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_reacher_easy agent=drqv2AE seed=12 experiment_id=9140 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_walker_walk agent=drqv2AE seed=10 experiment_id=9141 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_walker_walk agent=drqv2AE seed=11 experiment_id=9142 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_walker_walk agent=drqv2AE seed=12 experiment_id=9143 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &


# drqv2AE agent with camera distractions
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cup_catch agent=drqv2AE seed=10 experiment_id=9144 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cup_catch agent=drqv2AE seed=11 experiment_id=9145 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cup_catch agent=drqv2AE seed=12 experiment_id=9146 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cartpole_swingup agent=drqv2AE seed=10 experiment_id=9147 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cartpole_swingup agent=drqv2AE seed=11 experiment_id=9148 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cartpole_swingup agent=drqv2AE seed=12 experiment_id=9149 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cheetah_run agent=drqv2AE seed=10 experiment_id=9150 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cheetah_run agent=drqv2AE seed=11 experiment_id=9151 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cheetah_run agent=drqv2AE seed=12 experiment_id=9152 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_finger_spin agent=drqv2AE seed=10 experiment_id=9153 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_finger_spin agent=drqv2AE seed=11 experiment_id=9154 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_finger_spin agent=drqv2AE seed=12 experiment_id=9155 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_reacher_easy agent=drqv2AE seed=10 experiment_id=9156 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_reacher_easy agent=drqv2AE seed=11 experiment_id=9157 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_reacher_easy agent=drqv2AE seed=12 experiment_id=9158 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_walker_walk agent=drqv2AE seed=10 experiment_id=9159 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_walker_walk agent=drqv2AE seed=11 experiment_id=9160 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_walker_walk agent=drqv2AE seed=12 experiment_id=9161 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &


# V1_random_mask agent with background distractions
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cup_catch agent=V1_random_mask seed=10 experiment_id=9162 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cup_catch agent=V1_random_mask seed=11 experiment_id=9163 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cup_catch agent=V1_random_mask seed=12 experiment_id=9164 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cartpole_swingup agent=V1_random_mask seed=10 experiment_id=9165 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cartpole_swingup agent=V1_random_mask seed=11 experiment_id=9166 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cartpole_swingup agent=V1_random_mask seed=12 experiment_id=9167 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cheetah_run agent=V1_random_mask seed=10 experiment_id=9168 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cheetah_run agent=V1_random_mask seed=11 experiment_id=9169 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cheetah_run agent=V1_random_mask seed=12 experiment_id=9170 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_finger_spin agent=V1_random_mask seed=10 experiment_id=9171 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_finger_spin agent=V1_random_mask seed=11 experiment_id=9172 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_finger_spin agent=V1_random_mask seed=12 experiment_id=9173 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_reacher_easy agent=V1_random_mask seed=10 experiment_id=9174 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_reacher_easy agent=V1_random_mask seed=11 experiment_id=9175 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_reacher_easy agent=V1_random_mask seed=12 experiment_id=9176 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_walker_walk agent=V1_random_mask seed=10 experiment_id=9177 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_walker_walk agent=V1_random_mask seed=11 experiment_id=9178 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_walker_walk agent=V1_random_mask seed=12 experiment_id=9179 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path &


# V1_random_mask agent with color distractions
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cup_catch agent=V1_random_mask seed=10 experiment_id=9180 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cup_catch agent=V1_random_mask seed=11 experiment_id=9181 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cup_catch agent=V1_random_mask seed=12 experiment_id=9182 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cartpole_swingup agent=V1_random_mask seed=10 experiment_id=9183 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cartpole_swingup agent=V1_random_mask seed=11 experiment_id=9184 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cartpole_swingup agent=V1_random_mask seed=12 experiment_id=9185 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cheetah_run agent=V1_random_mask seed=10 experiment_id=9186 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cheetah_run agent=V1_random_mask seed=11 experiment_id=9187 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cheetah_run agent=V1_random_mask seed=12 experiment_id=9188 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_finger_spin agent=V1_random_mask seed=10 experiment_id=9189 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_finger_spin agent=V1_random_mask seed=11 experiment_id=9190 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_finger_spin agent=V1_random_mask seed=12 experiment_id=9191 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_reacher_easy agent=V1_random_mask seed=10 experiment_id=9192 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_reacher_easy agent=V1_random_mask seed=11 experiment_id=9193 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_reacher_easy agent=V1_random_mask seed=12 experiment_id=9194 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_walker_walk agent=V1_random_mask seed=10 experiment_id=9195 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_walker_walk agent=V1_random_mask seed=11 experiment_id=9196 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_walker_walk agent=V1_random_mask seed=12 experiment_id=9197 distraction.types=\[color\] distraction.dataset_path=bg_dataset_path &


# V1_random_mask agent with camera distractions
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cup_catch agent=V1_random_mask seed=10 experiment_id=9198 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cup_catch agent=V1_random_mask seed=11 experiment_id=9199 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cup_catch agent=V1_random_mask seed=12 experiment_id=9200 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cartpole_swingup agent=V1_random_mask seed=10 experiment_id=9201 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cartpole_swingup agent=V1_random_mask seed=11 experiment_id=9202 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cartpole_swingup agent=V1_random_mask seed=12 experiment_id=9203 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cheetah_run agent=V1_random_mask seed=10 experiment_id=9204 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cheetah_run agent=V1_random_mask seed=11 experiment_id=9205 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_cheetah_run agent=V1_random_mask seed=12 experiment_id=9206 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_finger_spin agent=V1_random_mask seed=10 experiment_id=9207 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_finger_spin agent=V1_random_mask seed=11 experiment_id=9208 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_finger_spin agent=V1_random_mask seed=12 experiment_id=9209 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_reacher_easy agent=V1_random_mask seed=10 experiment_id=9210 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_reacher_easy agent=V1_random_mask seed=11 experiment_id=9211 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_reacher_easy agent=V1_random_mask seed=12 experiment_id=9212 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_walker_walk agent=V1_random_mask seed=10 experiment_id=9213 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_walker_walk agent=V1_random_mask seed=11 experiment_id=9214 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=distracting_walker_walk agent=V1_random_mask seed=12 experiment_id=9215 distraction.types=\[camera\] distraction.dataset_path=bg_dataset_path &
