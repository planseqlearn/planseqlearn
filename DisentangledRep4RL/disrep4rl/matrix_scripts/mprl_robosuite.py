import os
import random
import time
env_names = [
    'robosuite_Lift',
    'robosuite_Door',
    'robosuite_PickPlaceBread',
    'robosuite_PickPlaceCan',
    'robosuite_PickPlaceCereal',
    'robosuite_PickPlaceMilk',
    'robosuite_NutAssemblyRound',
    'robosuite_NutAssemblySquare',
    # 'robosuite_NutAssembly',
    # 'robosuite_PickPlace',
    # 'robosuite_PickPlace',
]
num_seeds = 1
for idx, env_name in enumerate(env_names):
    valid_obj_names = ''
    if env_name == 'robosuite_PickPlace':
        if idx == 9:
            valid_obj_names = '["Milk", "Cereal"]'
        if idx == 10:
            valid_obj_names = '["Can", "Bread"]'
        
    for _ in range(num_seeds):
        seed = random.randint(0, 100000000)
        run_name = f"drqv2_mprl_{env_name}_{valid_obj_names}"
        cmd = f"python disrep4rl/train.py agent=drqv2 use_wandb=True seed={seed} debug=True save_video=True num_train_frames=1000000 eval_every_frames=10000 num_eval_episodes=10 wandb.project_name=mprl_paper_results wandb.run_name={run_name} task={env_name} valid_obj_names={valid_obj_names} mprl=True path_length=50 action_repeat=1 experiment_id={run_name} matrix=True"
        os.system(cmd)
        time.sleep(5)