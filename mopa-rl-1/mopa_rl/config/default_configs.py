LIFT_CONFIG = dict(
    id="SawyerLift-v0",
    action_repeat=5, 
    camera_name='agentview', #camera_name='visview', 
    contact_threshold=-0.002, 
    ctrl_reward_coef=0, 
    debug=False, 
    distance_threshold=0.06, 
    frame_dt=0.15, 
    frame_skip=1, 
    joint_margin=0.001, 
    kd=8.0, ki=0.0, 
    kp=40.0, 
    max_episode_steps=250, 
    range=0.1, 
    reward_type='dense', 
    screen_height=84, 
    screen_width=84, 
    seed=1234, 
    simple_planner_range=0.05, 
    simple_planner_timelimit=0.05, 
    step_size=0.02, 
    success_reward=1.0, # was previously 150.0 
    timelimit=1.0, 
    use_robot_indicator=True, 
    use_target_robot_indicator=True,
    ik_target="grip_site",
    use_ik_target=True,
    action_range=0.05, # used to be 1 -> can change to 
    ac_scale=0.05, # used to be 1
)

LIFT_OBSTACLE_CONFIG = dict(
    id="SawyerLiftObstacle-v0",
    action_repeat=5, 
    camera_name='visview', 
    contact_threshold=-0.002, 
    ctrl_reward_coef=0, 
    debug=False, 
    distance_threshold=0.06, 
    frame_dt=0.15, 
    frame_skip=1, 
    joint_margin=0.001, 
    kd=8.0, ki=0.0, 
    kp=40.0, 
    max_episode_steps=250, 
    range=0.1, 
    reward_type='dense', 
    screen_height=84, 
    screen_width=84, 
    seed=1234, 
    simple_planner_range=0.05, 
    simple_planner_timelimit=0.05, 
    step_size=0.02, 
    success_reward=1.0, # was previously 150.0 
    timelimit=1.0, 
    use_robot_indicator=True, 
    use_target_robot_indicator=True,
    ik_target="grip_site",
    use_ik_target=True,
    action_range=0.05,
    ac_scale=0.05,
)


ASSEMBLY_OBSTACLE_CONFIG = dict(
    id="SawyerAssemblyObstacle-v0",
    action_repeat=5, 
    camera_name='zoomview', 
    contact_threshold=-0.002, 
    ctrl_reward_coef=0, 
    debug=False, 
    distance_threshold=0.06, 
    frame_dt=0.15, 
    frame_skip=1, 
    joint_margin=0.001, 
    kd=8.0, ki=0.0, 
    kp=40.0, 
    max_episode_steps=250, 
    range=0.1, 
    reward_type='dense', 
    screen_height=84, 
    screen_width=84, 
    seed=1234, 
    simple_planner_range=0.05, 
    simple_planner_timelimit=0.05, 
    step_size=0.02, 
    success_reward=1.0, 
    timelimit=1.0, 
    use_robot_indicator=True, 
    use_target_robot_indicator=True,
    ik_target="grip_site",
    use_ik_target=True,
    action_range=0.05, 
    ac_scale=0.05,
)

PUSHER_OBSTACLE_CONFIG = dict(
    id="SawyerPushObstacle-v0",
    action_repeat=5, 
    camera_name='visview', 
    contact_threshold=-0.002, 
    ctrl_reward_coef=0, 
    debug=False, 
    distance_threshold=0.06, 
    frame_dt=0.15, 
    frame_skip=1, 
    joint_margin=0.001, 
    kd=8.0, ki=0.0, 
    kp=40.0, 
    max_episode_steps=250, 
    range=0.1, 
    reward_type='dense', 
    screen_height=84, 
    screen_width=84, 
    seed=1234, 
    simple_planner_range=0.05, 
    simple_planner_timelimit=0.05, 
    step_size=0.02, 
    success_reward=150.0, 
    timelimit=1.0, 
    use_robot_indicator=True, 
    use_target_robot_indicator=True,
    ik_target="grip_site",
    use_ik_target=True,
    action_range=0.05, 
    ac_scale=0.05,
)
