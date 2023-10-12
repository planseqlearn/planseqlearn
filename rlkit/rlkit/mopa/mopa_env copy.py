import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
from gym import spaces
import collections
import mujoco_py  # added to fix build import error
from robosuite.utils.transform_utils import (
    convert_quat,
    euler2mat,
    mat2pose,
    mat2quat,
    pose2mat,
    quat2mat,
    mat2quat,
    quat_conjugate,
    quat_multiply,
)
import time
from mopa_rl.env.inverse_kinematics import (
    qpos_from_site_pose_sampling,
    qpos_from_site_pose,
)
import robosuite.utils.transform_utils as T
from mopa_rl.config.default_configs import (
    LIFT_OBSTACLE_CONFIG,
    LIFT_CONFIG,
    ASSEMBLY_OBSTACLE_CONFIG,
    PUSHER_OBSTACLE_CONFIG,
)

try:
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    import sys
    from os.path import abspath, dirname, join

    sys.path.insert(0, join(dirname(dirname(abspath(__file__))), "py-bindings"))
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou

LIFT_ROBOT_BODIES = [
    "base",
    "controller_box",
    "pedestal_feet",
    "torso",
    "pedestal",
    "right_arm_base_link",
    "right_l0",
    "head",
    "screen",
    "right_l1",
    "right_l2",
    "right_l3",
    "right_l4",
    "right_l5",
    "right_l6",
    "right_ee_attchment",
    "clawGripper",
    "right_gripper_base",
    "right_gripper",
    "rightclaw",
    "r_gripper_l_finger_tip",
    "leftclaw",
    "r_gripper_r_finger_tip",
    "base_indicator",
    "right_arm_base_link_indicator",
    "right_l0_indicator",
    "head_indicator",
    "screen_indicator",
    "right_l1_indicator",
    "right_l2_indicator",
    "right_l3_indicator",
    "right_l4_indicator",
    "right_l5_indicator",
    "right_l6_indicator",
    "right_ee_attchment_indicator",
    "clawGripper_indicator",
    "right_gripper_base_indicator",
    "rightclaw_indicator",
    "r_gripper_l_finger_tip_indicator",
    "leftclaw_indicator",
    "r_gripper_r_finger_tip_indicator",
    "base_target",
    "right_arm_base_link_target",
    "right_l0_target",
    "right_l1_target",
    "right_l2_target",
    "right_l3_target",
    "right_l4_target",
    "right_l5_target",
    "right_l6_target",
    "right_ee_attchment_target",
    "clawGripper_target",
    "right_gripper_base_target",
    "rightclaw_target",
    "r_gripper_l_finger_tip_target",
    "leftclaw_target",
    "r_gripper_r_finger_tip_target",
]


def save_img(env, filename="test.png"):
    frame = env.render("rgb_array")
    plt.imshow(frame)
    plt.savefig(filename)


def get_object_name(env_name):
    if (
        env_name == "SawyerLift-v0"
        or env_name == "SawyerLiftObstacle-v0"
        or env_name == "SawyerPushObstacle-v0"
    ):
        return "cube"
    elif env_name == "SawyerAssemblyObstacle-v0":
        return "0_part0"
    else:
        raise NotImplementedError


def get_site_pose(env, name):
    """
    Gets pose of site.
    Args:
        env: Gym environment
        name: name of site in sim
    Returns:
        (pos, orn) tuple where pos is xyz location of site, orn
            is (4,) numpy.ndarray corresponding to quat in xyzw format
    """
    xpos = env.sim.data.get_site_xpos(name)[: len(env.min_world_size)].copy()
    model = env.sim.model
    xquat = mat2quat(env.sim.data.get_site_xmat(name).copy())
    return xpos, xquat


def get_object_pose(env, name):
    """
    Gets pose of desired object.
    Args:
        env: Gym environment
        name: name of manipulation object from body_names in sim
    Returns:
        (7,) numpy.ndarray corresponding to position and quaternion,
            where quaternion is in xyzw format
    """
    start = env.sim.model.body_jntadr[env.sim.model.body_name2id(name)]
    xpos = env.sim.data.qpos[start : start + 3].copy()
    xquat = env.sim.data.qpos[start + 3 : start + 7].copy()
    xquat = convert_quat(xquat, to="xyzw")
    return np.concatenate((xpos, xquat))


def set_object_pose(env, name, new_xpos, new_xquat):
    """
    Gets pose of desired object.
    Args:
        env: Gym environment
        name: name of manipulation object from body_names in sim
        new_xpos: (3,) numpy.ndarray of new xyz coordinates
        new_xquat: (4,) numpy.ndarary of new quaternion, in xyzw format
    Returns:
        None
    """
    start = env.sim.model.body_jntadr[env.sim.model.body_name2id(name)]
    new_xquat = convert_quat(new_xquat, to="wxyz")
    env.sim.data.qpos[start : start + 3] = new_xpos
    env.sim.data.qpos[start + 3 : start + 7] = new_xquat


def check_grasp(env, name):
    """
    Checks grasp of object in environment.
    Args:
        env: Gym environment
        name: name of environment
    Returns:
        boolean corresponding to grasped object
    """
    if name == "SawyerLift-v0" or name == "SawyerLiftObstacle-v0":
        touch_left_finger = False
        touch_right_finger = False
        for i in range(env.sim.data.ncon):
            c = env.sim.data.contact[i]
            if c.geom1 == env.cube_geom_id:
                if c.geom2 in env.l_finger_geom_ids:
                    touch_left_finger = True
                if c.geom2 in env.r_finger_geom_ids:
                    touch_right_finger = True
            elif c.geom2 == env.cube_geom_id:
                if c.geom1 in env.l_finger_geom_ids:
                    touch_left_finger = True
                if c.geom1 in env.r_finger_geom_ids:
                    touch_right_finger = True
        return touch_left_finger and touch_right_finger
    if name == "SawyerAssemblyObstacle-v0":
        return False  # there is no sense of grasping in the assembly environment so return false
    if name == "SawyerPushObstacle-v0":
        return False
    else:
        raise NotImplementedError


# TODO do full collision checking for Assembly
def check_collisions(env, allowed_collision_pairs, env_name, verbose=False):
    mjcontacts = env.sim.data.contact
    ncon = env.sim.data.ncon
    for i in range(ncon):
        ct = mjcontacts[i]
        ct1 = ct.geom1
        ct2 = ct.geom2
        b1 = env.sim.model.geom_bodyid[ct1]
        b2 = env.sim.model.geom_bodyid[ct2]
        bn1 = env.sim.model.body_id2name(b1)
        bn2 = env.sim.model.body_id2name(b2)
        if verbose:
            print(f"ct1:{ct1} ct2:{ct2} b1:{bn1} b2:{bn2}")
        # robot bodies checking allows robot to collide with itself
        # useful for going up when grasping
        if env_name == "SawyerLift-v0" or env_name == "SawyerLiftObstacle-v0":
            if (
                ((bn1 in LIFT_ROBOT_BODIES) and (bn2 in LIFT_ROBOT_BODIES))
                or ((bn1 in LIFT_ROBOT_BODIES) and (bn2 == "cube"))
                or ((bn2 in LIFT_ROBOT_BODIES) and (bn1 == "cube"))
                or ((ct1, ct2) in allowed_collision_pairs)
                or ((ct2, ct1) in allowed_collision_pairs)
            ):
                continue
            else:
                return True
        elif env_name == "SawyerAssemblyObstacle-v0":
            if ((ct1, ct2) not in allowed_collision_pairs) and (
                (ct2, ct1) not in allowed_collision_pairs
            ):
                if verbose:
                    print(f"Case")
                return True
    return False


def set_robot_based_on_ee_pos(
    env,
    ac,
    ik_env,
    qpos,
    qvel,
    is_grasped,
    target_object,
    config,
    allowed_collision_pairs,
    env_name="SawyerLift-v0",
    return_angles=False,
):
    """
    Takes in action in the format of desired delta in orientation and position
    and teleports there.
    Args:
        env: Gym environment
        ac: OrderedDict - should have keys 'default' and optionally 'quat'
            corresponding to target xyz and quat (where quat is in wxyz format)
        ik_env: Gym environment - copy of env where ik algorithm is run
        qpos: canonical pose to reset to when running IK
        qvel: same as above
        is_grasped: whether object is grasped or not
        config: config file of environment
    Returns:
        (success, err_norm), where success is if ik is successful, err_norm
            is how far we are from desired target
    """
    start = time.time()
    # keep track of gripper pos, etc
    gripper_qpos = env.sim.data.qpos[env.ref_gripper_joint_pos_indexes].copy()
    gripper_qvel = env.sim.data.qvel[env.ref_gripper_joint_pos_indexes].copy()
    old_eef_xpos, old_eef_xquat = get_site_pose(env, config["ik_target"])
    try:
        object_pose = get_object_pose(env, target_object).copy()
    except:
        pass
    # target_cart = np.clip(
    #     env.sim.data.get_site_xpos(config["ik_target"])[: len(env.min_world_size)]
    #     + ac["default"],
    #     env.min_world_size,
    #     env.max_world_size,
    # )
    target_cart = np.clip(
        ac["default"],
        env.min_world_size,
        env.max_world_size,
    )
    if "quat" in ac.keys():
        target_quat = ac["quat"]
    else:
        target_quat = None
    ik_env.set_state(qpos, qvel)  # check this doesn't break anything
    # print(target_cart, target_quat)
    result = qpos_from_site_pose(
        ik_env,
        config["ik_target"],
        target_pos=target_cart,
        target_quat=target_quat,
        rot_weight=2.0,
        joint_names=env.robot_joints,
        max_steps=100,
        tol=1e-2,
    )
    if result.success == False or check_collisions(
        ik_env, allowed_collision_pairs, env_name
    ):
        return (
            result.success
            and not check_collisions(ik_env, allowed_collision_pairs, env_name),
            result.err_norm,
        )
    if return_angles:
        return ik_env.sim.data.qpos.copy()[:7]
    # set state here
    env.set_state(ik_env.sim.data.qpos.copy(), ik_env.sim.data.qvel.copy())

    if is_grasped:
        env.sim.data.qpos[env.ref_gripper_joint_pos_indexes] = gripper_qpos
        env.sim.data.qvel[env.ref_gripper_joint_pos_indexes] = gripper_qvel

        # compute transform between new and old
        ee_old_mat = pose2mat((old_eef_xpos, old_eef_xquat))
        new_eef_xpos, new_eef_xquat = get_site_pose(env, config["ik_target"])
        ee_new_mat = pose2mat((new_eef_xpos, new_eef_xquat))
        transform = ee_new_mat @ np.linalg.inv(ee_old_mat)

        # get new object pose
        new_object_pose = mat2pose(
            np.dot(transform, pose2mat((object_pose[:3], object_pose[3:])))
        )
        set_object_pose(env, target_object, new_object_pose[0], new_object_pose[1])
        env.sim.forward()
    return result.success, result.err_norm


def set_robot_based_on_joint(
    env,
    ac,
    ik_env,
    qpos,
    qvel,
    is_grasped,
    target_object,
    config,
    allowed_collision_pairs,
    env_name="SawyerLift-v0",
    return_angles=False,
):
    """
    Takes in action in the format of desired delta in orientation and position
    and teleports there.
    Args:
        env: Gym environment
        ac: OrderedDict - should have keys 'default' and optionally 'quat'
            corresponding to target xyz and quat (where quat is in wxyz format)
        ik_env: Gym environment - copy of env where ik algorithm is run
        qpos: canonical pose to reset to when running IK
        qvel: same as above
        is_grasped: whether object is grasped or not
        config: config file of environment
    Returns:
        (success, err_norm), where success is if ik is successful, err_norm
            is how far we are from desired target
    """
    start = time.time()
    # keep track of gripper pos, etc
    gripper_qpos = env.sim.data.qpos[env.ref_gripper_joint_pos_indexes].copy()
    gripper_qvel = env.sim.data.qvel[env.ref_gripper_joint_pos_indexes].copy()
    old_eef_xpos, old_eef_xquat = get_site_pose(env, config["ik_target"])
    try:
        object_pose = get_object_pose(env, target_object).copy()
    except:
        pass
    # set state here
    env.set_state(np.concatenate((ac, env.sim.data.qpos[7:])), env.sim.data.qvel.copy())

    if is_grasped:
        env.sim.data.qpos[env.ref_gripper_joint_pos_indexes] = gripper_qpos
        env.sim.data.qvel[env.ref_gripper_joint_pos_indexes] = gripper_qvel

        # compute transform between new and old
        ee_old_mat = pose2mat((old_eef_xpos, old_eef_xquat))
        new_eef_xpos, new_eef_xquat = get_site_pose(env, config["ik_target"])
        ee_new_mat = pose2mat((new_eef_xpos, new_eef_xquat))
        transform = ee_new_mat @ np.linalg.inv(ee_old_mat)

        # get new object pose
        new_object_pose = mat2pose(
            np.dot(transform, pose2mat((object_pose[:3], object_pose[3:])))
        )
        set_object_pose(env, target_object, new_object_pose[0], new_object_pose[1])
        env.sim.forward()


def mp_to_point(
    env,
    ac,
    ik_env,
    qpos,
    qvel,
    allowed_collision_pairs,
    is_grasped=False,
    ignore_object_collision=False,
    planning_time=1,
    get_intermediate_frames=False,
    backtrack_movement_fraction=0.01,
    env_name="SawyerLift-v0",
    config=LIFT_CONFIG,
    mp_env=None,
):
    qpos_curr = env.sim.data.qpos.copy()
    qvel_curr = env.sim.data.qvel.copy()

    # consider original xpos and xquat
    og_eef_xpos, og_eef_xquat = get_site_pose(env, "grip_site")
    # og_eef_xpos = np.zeros(3, dtype=np.float64) # og_eef_xpos.astype(np.float64)
    og_eef_xquat /= np.linalg.norm(og_eef_xquat)
    # convert quats
    og_eef_xquat = convert_quat(og_eef_xquat, to="wxyz").astype(np.float64)
    if "quat" in ac.keys():
        ac["quat"] = ac["quat"].astype(np.float64)
        # ac["quat"] = convert_quat(ac["quat"], to="wxyz") -> ignore for now, already in wxyz format
    else:
        ac["quat"] = og_eef_xquat

    # think about case where we don't have ac["quat"] -> maybe can just fix this for
    # the pusher task
    def isStateValid(state):
        # get pos
        pos = np.array([state.getX(), state.getY(), state.getZ()])
        # get quat
        quat = np.array(
            [
                state.rotation().w,
                state.rotation().x,
                state.rotation().y,
                state.rotation().z,
            ]
        )
        if all(pos == og_eef_xpos) and all(quat == og_eef_xquat):
            # start state is always valid.
            return True
        else:
            # define action correctly in terms of state
            action = collections.OrderedDict()
            action[
                "default"
            ] = pos  # pos should be a delta to the goal, not the goal itself
            action["quat"] = quat
            result, err_norm = set_robot_based_on_ee_pos(
                env,
                action,
                ik_env,
                qpos,
                qvel,
                is_grasped,
                get_object_name(env_name),
                config,
                allowed_collision_pairs,
                env_name,
            )
            # save_img(ik_env, "test.png")
            return result  # note result also keeps track of whether there was a collision or not

    # create an SE3 state space
    space = ob.SE3StateSpace()

    # set lower and upper bounds
    bounds = ob.RealVectorBounds(3)
    # compare bounds to start state
    bounds_low = env.min_world_size
    bounds_high = env.max_world_size
    # bounds_low = np.minimum(bounds_low, og_eef_xpos)
    # bounds_high = np.maximum(bounds_high, og_eef_xpos)
    bounds_low = np.array([-5.0, -5.0, -5.0])
    bounds_high = np.array([5.0, 5.0, 5.0])

    bounds.setLow(0, bounds_low[0])
    bounds.setLow(1, bounds_low[1])
    bounds.setLow(2, bounds_low[2])
    bounds.setHigh(0, bounds_high[0])
    bounds.setHigh(1, bounds_high[1])
    bounds.setHigh(2, bounds_high[2])
    space.setBounds(bounds)

    # construct an instance of space information from this state space
    si = ob.SpaceInformation(space)
    # set state validity checking for this space
    si.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))
    si.setStateValidityCheckingResolution(0.001)  # default of 0.01 is too coarse
    # divide by norm again to make sure
    og_eef_xquat /= np.linalg.norm(og_eef_xquat)
    ac["quat"] /= np.linalg.norm(ac["quat"])
    # create a random start state
    start = ob.State(space)
    start().setXYZ(*og_eef_xpos)
    start().rotation().w = og_eef_xquat[0]
    start().rotation().x = og_eef_xquat[1]
    start().rotation().y = og_eef_xquat[2]
    start().rotation().z = og_eef_xquat[3]

    goal = ob.State(space)
    goal().setXYZ(*ac["default"])
    goal().rotation().w = ac["quat"][0]
    goal().rotation().x = ac["quat"][1]
    goal().rotation().y = ac["quat"][2]
    goal().rotation().z = ac["quat"][3]
    goal_valid = isStateValid(goal())
    print(f"Start state: {[start().getX(), start().getY(), start().getZ()]}")
    print(f"Start satisfies bounds: {space.satisfiesBounds(start())}")
    print(f"Goal satisfies bounds: {space.satisfiesBounds(goal())}")
    print(f"Start state is valid: {isStateValid(start())}")
    print(f"Goal state is valid: {isStateValid(goal())}")
    # implement if not goal_valid but ignoring for now
    # create a problem instance
    pdef = ob.ProblemDefinition(si)
    # set the start and goal states
    pdef.setStartAndGoalStates(start, goal)
    # create a planner for the defined space
    planner = og.RRTConnect(si)
    # set the problem we are trying to solve for the planner
    planner.setProblemDefinition(pdef)
    planner.setRange(0.05)
    # perform setup steps for the planner
    planner.setup()
    # attempt to solve the problem within planning_time seconds of planning time
    intermediate_frames = []
    solved = planner.solve(planning_time)
    print(f"Solved??: {solved}")
    if solved:
        path = pdef.getSolutionPath()
        success = og.PathSimplifier(si).simplify(path, 1.0)
        print(f"Simplified path")
        converted_path = []
        for s, state in enumerate(path.getStates()):
            new_state = [
                state.getX(),
                state.getY(),
                state.getZ(),
                state.rotation().w,
                state.rotation().x,
                state.rotation().y,
                state.rotation().z,
            ]
            converted_path.append(new_state)

        # set to every state in the path and see what it looks like:
        # waypoint_images = []
        # waypoint_masks = []
        # for i, state in enumerate(converted_path):
        #     ac = collections.OrderedDict()
        #     ac["default"] = state[:3]
        #     ac["quat"] = state[3:]
        #     set_robot_based_on_ee_pos(
        #             env,
        #             ac,
        #             ik_env,
        #             qpos.copy(),
        #             qvel.copy(),
        #             is_grasped,
        #             get_object_name(env_name),
        #             config,
        #             allowed_collision_pairs,
        #             env_name,
        #         )
        #     im = mp_env.get_image()
        #     # cv2.imwrite("test_{i}.png".format(i=i), im)
        #     sim = env.sim
        #     segmentation_map = CU.get_camera_segmentation(
        #         camera_name="frontview",
        #         camera_width=960,
        #         camera_height=540,
        #         sim=sim,
        #     )
        #     # get robot segmentation mask
        #     geom_ids = np.unique(segmentation_map[:, :, 1])
        #     robot_ids = []
        #     for geom_id in geom_ids:
        #         if geom_id > -1:
        #             geom_name = sim.model.geom_id2name(geom_id)
        #             if geom_name is None or geom_name.startswith("Visual"):
        #                 continue
        #             if geom_name.startswith("right_") or geom_name.endswith("claw"):
        #                 robot_ids.append(geom_id)
        #     robot_mask = np.expand_dims(np.any(
        #         [segmentation_map[:, :, 1] == robot_id for robot_id in robot_ids], axis=0
        #     ), -1)
        #     waypoint_masks.append(robot_mask)
        #     # cv2.imwrite('masked_test_{i}.png'.format(i=i), robot_mask*im)
        #     waypoint_images.append(robot_mask*im)
        env.reset()
        env.sim.data.qpos[:] = qpos_curr.copy()
        env.sim.data.qvel[:] = qvel_curr.copy()
        env.sim.forward()
        # mp_env.set_robot_color(np.array([0.1, 0.3, 0.7, 1.0]))
        i = 0
        for state_idx, state in enumerate(converted_path):
            i += 1
            ac = collections.OrderedDict()
            ac["default"] = state[:3]
            ac["quat"] = state[3:]
            target_angles = set_robot_based_on_ee_pos(
                env,
                ac,
                ik_env,
                qpos.copy(),
                qvel.copy(),
                is_grasped,
                get_object_name(env_name),
                config,
                allowed_collision_pairs,
                env_name,
                return_angles=True,
            )
            start_angles = env.sim.data.qpos[:7].copy()
            for step in range(25):
                ac = collections.OrderedDict()
                eef_xpos, eef_xquat = get_site_pose(env, "grip_site")
                # ac["default"] = (state[:3] - eef_xpos)*step/50 + eef_xpos
                # # ac["default"] = state[:3]
                # ac["quat"] = (state[3:] - eef_xquat) * step / 50 + eef_xquat
                # set_robot_based_on_ee_pos(
                #     env,
                #     ac,
                #     ik_env,
                #     qpos.copy(),
                #     qvel.copy(),
                #     is_grasped,
                #     get_object_name(env_name),
                #     config,
                #     allowed_collision_pairs,
                #     env_name
                # )
                ac = (target_angles - start_angles) * step / 25 + start_angles
                set_robot_based_on_joint(
                    env,
                    ac,
                    ik_env,
                    qpos.copy(),
                    qvel.copy(),
                    is_grasped,
                    get_object_name(env_name),
                    config,
                    allowed_collision_pairs,
                    env_name,
                )

                # if get_intermediate_frames:
                #     im = mp_env.get_image()
                #     # if state_idx > 0:
                #     #     robot_mask = waypoint_masks[state_idx]
                #     #     im = 0.5 * (im * robot_mask) + 0.5 * waypoint_images[state_idx] + im * (1 - robot_mask)
                #     # add_text(im, "Planner", (1, 10), 0.5, (0, 255, 0))
                #     intermediate_frames.append(im)
            # converted_ac = cart2joint_ac(
            #     env,
            #     ik_env,
            #     ac,
            #     qpos,
            #     qvel,
            #     config,
            # )
            # env.step
        # mp_env.reset_robot_color()
        env.intermediate_frames = intermediate_frames


def backtracking_search_from_goal(
    env,
    ik_env,
    ac,
    is_grasped,
    qpos,
    qvel,
    target_object,
    config,
    movement_fraction,
    allowed_collision_pairs,
    env_name="SawyerLift-v0",
):
    """
    This function takes in a desired action delta and (optionally) a desired orientation delta
    for the current environment
    """
    xyz_pos = ac["default"].copy()
    new_ac = collections.OrderedDict()
    new_ac["default"] = xyz_pos.copy()
    max_iters = int(1 / movement_fraction)
    iters = 0
    while iters < max_iters:
        success, _ = set_robot_based_on_ee_pos(
            env,
            new_ac,
            ik_env,
            qpos,
            qvel,
            is_grasped,
            target_object,
            config,
            allowed_collision_pairs,
            env_name,
        )
        # save image from ik_env
        if success:  # here, we have already set the position in set_robot
            break
        iters += 1
        new_ac["default"] = (1 - iters * movement_fraction) * xyz_pos


def cart2joint_ac(
    env,
    ik_env,
    ac,
    qpos,
    qvel,
    config,
):
    curr_qpos = env.sim.data.qpos.copy()
    target_cart = np.clip(
        env.sim.data.get_site_xpos(config["ik_target"])[: len(env.min_world_size)]
        + config["action_range"] * ac["default"],
        env.min_world_size,
        env.max_world_size,
    )
    if len(env.min_world_size) == 2:
        target_cart = np.concatenate(
            (
                target_cart,
                np.array([env.sim.data.get_site_xpos(config["ik_target"])[2]]),
            )
        )
    if "quat" in ac.keys():
        target_quat = ac["quat"]
    else:
        target_quat = None
    ik_env.set_state(curr_qpos.copy(), env.data.qvel.copy())
    result = qpos_from_site_pose(
        ik_env,
        config["ik_target"],
        target_pos=target_cart,
        target_quat=target_quat,
        rot_weight=2.0,
        joint_names=env.robot_joints,
        max_steps=100,
        tol=1e-2,
    )
    target_qpos = env.sim.data.qpos.copy()
    target_qpos[env.ref_joint_pos_indexes] = result.qpos[
        env.ref_joint_pos_indexes
    ].copy()
    pre_converted_ac = (
        target_qpos[env.ref_joint_pos_indexes] - curr_qpos[env.ref_joint_pos_indexes]
    ) / env._ac_scale
    if "gripper" in ac.keys():
        pre_converted_ac = np.concatenate((pre_converted_ac, ac["gripper"]))
    converted_ac = collections.OrderedDict([("default", pre_converted_ac)])
    return converted_ac


################## VISION PIPELINE ##################
import robosuite.utils.camera_utils as CU


def get_camera_depth(sim, camera_name, camera_height, camera_width):
    """
    Obtains depth image.

    Args:
        sim (MjSim): simulator instance
        camera_name (str): name of camera
        camera_height (int): height of camera images in pixels
        camera_width (int): width of camera images in pixels
    Return:
        im (np.array): the depth image b/w 0 and 1
    """
    return sim.render(
        camera_name=camera_name, height=camera_height, width=camera_width, depth=True
    )[1][::-1]


def get_object_pose_from_seg(
    env, object_string, camera_name, camera_width, camera_height, sim
):
    segmentation_map = CU.get_camera_segmentation(
        camera_name=camera_name,
        camera_width=camera_width,
        camera_height=camera_height,
        sim=sim,
    )
    obj_id = sim.model.geom_name2id(object_string)
    obj_mask = segmentation_map == obj_id
    depth_map = get_camera_depth(
        camera_name=camera_name,
        camera_width=camera_width,
        camera_height=camera_height,
        sim=sim,
    )
    depth_map = np.expand_dims(
        CU.get_real_depth_map(sim=env.sim, depth_map=depth_map), -1
    )
    world_to_camera = CU.get_camera_transform_matrix(
        camera_name=camera_name,
        camera_width=camera_width,
        camera_height=camera_height,
        sim=sim,
    )
    camera_to_world = np.linalg.inv(world_to_camera)
    obj_pointcloud = CU.transform_from_pixels_to_world(
        pixels=np.argwhere(obj_mask),
        depth_map=depth_map[..., 0],
        camera_to_world_transform=camera_to_world,
    )
    assert len(obj_pointcloud) > 0
    return np.mean(obj_pointcloud, axis=0)


################## VISION PIPELINE ##################


class MoPAMPEnv:
    def __init__(
        self,
        name,
        env,
        ik_env,
        # motion_planning
        plan_to_learned_goals=False,
        planner_only_actions=False,
        num_ll_actions_per_hl_action=1,
        planner_command_orientation=False,
        # teleporting
        vertical_displacement=0.07,
        teleport_instead_of_mp=True,
        teleport_on_grasp=True,
        # vision
        use_vision_pose_estimation=False,
        # config for mopa-rl
        config=None,
        horizon=50,
        ignore_done=False,
        no_mprl=False,  # if no mprl, then only run regular actions
        mprl=True,
    ):
        self._wrapped_env = env
        self.ik_env = ik_env
        # add config abilities later
        self.name = name
        # motion_planning
        self.plan_to_learned_goals = plan_to_learned_goals
        self.planner_only_actions = planner_only_actions
        # teleporting
        self.vertical_displacement = vertical_displacement
        self.planner_command_orientation = planner_command_orientation
        self.teleport_instead_of_mp = teleport_instead_of_mp
        self.teleport_on_grasp = teleport_on_grasp
        self.config = config
        self.num_ll_actions_per_hl_action = num_ll_actions_per_hl_action
        # step counts
        self.ep_step_ctr = 0
        self.high_level_step = 0
        self.num_failed_solves = 0
        self.num_steps = 0
        self.current_ll_policy_steps = 0
        self.horizon = horizon
        self.ignore_done = ignore_done
        # more planner stuff
        self.take_planner_step = True
        self.learn_residual = False
        self.no_mprl = no_mprl
        self.use_vision_pose_estimation = use_vision_pose_estimation
        # collision checking
        self.allowed_collision_pairs = []
        for manipulation_geom_id in self._wrapped_env.manipulation_geom_ids:
            for geom_id in self._wrapped_env.static_geom_ids:
                self.allowed_collision_pairs.append((manipulation_geom_id, geom_id))
        if self.name == "SawyerLift-v0" or self.name == "SawyerLiftObstacle-v0":
            for manipulation_geom_id in self._wrapped_env.manipulation_geom_ids:
                for lf in self._wrapped_env.left_finger_geoms:
                    self.allowed_collision_pairs.append(
                        (
                            manipulation_geom_id,
                            self._wrapped_env.sim.model.geom_name2id(lf),
                        )
                    )
            for manipulation_geom_id in self._wrapped_env.manipulation_geom_ids:
                for rf in self._wrapped_env.right_finger_geoms:
                    self.allowed_collision_pairs.append(
                        (
                            manipulation_geom_id,
                            self._wrapped_env.sim.model.geom_name2id(rf),
                        )
                    )
        if self.name == "SawyerLift-v0":
            config = LIFT_CONFIG
            if mprl:
                config["camera_name"] = "eye_in_hand"
        elif self.name == "SawyerLiftObstacle-v0":
            config = LIFT_OBSTACLE_CONFIG
            if mprl:
                config["camera_name"] = "eye_in_hand"
            # if is_eval:
            #     config["camera_name"] = "visview"
        elif self.name == "SawyerAssemblyObstacle-v0":
            config = ASSEMBLY_OBSTACLE_CONFIG
            if mprl:
                config["camera_name"] = "eye_in_hand"
        elif self.name == "SawyerPushObstacle-v0":
            config = PUSHER_OBSTACLE_CONFIG
            if mprl:
                config["camera_name"] = "eye_in_hand"
        self.config = config
        # self.robot_bodies = [
        #     "right_arm_base_link",
        #     "right_l0", "right_l1", "right_l2", "right_l3", "right_l4",
        #     "right_l5", "right_l6", "right_ee_attchment", "clawGripper", "rightclaw", "leftclaw"
        # ]
        # self.robot_body_ids, self.robot_geom_ids = self.get_body_geom_ids_from_robot_bodies()
        # self.original_colors = [env.sim.model.geom_rgba[idx].copy() for idx in self.robot_geom_ids]

    def get_body_geom_ids_from_robot_bodies(self):
        body_ids = [
            self._wrapped_env.sim.model.body_name2id(body) for body in self.robot_bodies
        ]
        geom_ids = []
        for geom_id, body_id in enumerate(self._wrapped_env.sim.model.geom_bodyid):
            if body_id in body_ids:
                geom_ids.append(geom_id)
        return body_ids, geom_ids

    def set_robot_color(self, colors):
        if type(colors) is np.ndarray:
            colors = [colors] * len(self.robot_geom_ids)
        for idx, geom_id in enumerate(self.robot_geom_ids):
            self._wrapped_env.sim.model.geom_rgba[geom_id] = colors[idx]
        self._wrapped_env.sim.forward()

    def reset_robot_color(self):
        self.set_robot_color(self.original_colors)
        self._wrapped_env.sim.forward()

    @property
    def observation_space(self):
        obs_space = self._wrapped_env.observation_space
        low = np.array([])
        high = np.array([])
        for k in obs_space.spaces.keys():
            low = np.concatenate((low, obs_space[k].low))
            high = np.concatenate((high, obs_space[k].high))
        return spaces.Box(
            low=low,
            high=high,
        )

    @property
    def action_space(self):
        low = np.array([-np.inf for _ in range(7)])
        high = np.array([np.inf for _ in range(7)])
        return spaces.Box(
            low=low,
            high=high,
        )

    def _get_observation(self):
        obs = self._wrapped_env._get_obs()
        observation = np.array([])
        for k in obs.keys():
            observation = np.concatenate((observation, obs[k]))
        return observation

    def _convert_observation(self, obs):
        observation = np.array([])
        for k in obs.keys():
            observation = np.concatenate((observation, obs[k].copy()))
        return observation

    def get_init_target_pos(self):
        qpos, qvel = (
            self._wrapped_env.sim.data.qpos.copy(),
            self._wrapped_env.sim.data.qvel.copy(),
        )
        if self.name == "SawyerLift-v0" or self.name == "SawyerLiftObstacle-v0":
            # get cube position
            if self.use_vision_pose_estimation:
                cube_pos = get_object_pose_from_seg(
                    env=self._wrapped_env,
                    object_string="cube",
                    camera_name="topview",
                    camera_width=500,
                    camera_height=500,
                    sim=self._wrapped_env.sim,
                )
                cube_pos += np.array([0.0, 0.0, 0.07])  # double check with 0.05, etc
            else:
                cube_pos = get_object_pose(self._wrapped_env, "cube")[
                    :3
                ].copy() + np.array([0.0, 0.00, self.vertical_displacement])
            quat = np.array([-0.1268922, 0.21528646, 0.96422245, -0.08846001])
            quat /= np.linalg.norm(quat)
            # get gripper position
            ac = collections.OrderedDict()
            ac["default"] = cube_pos
            ac["quat"] = quat
            # result, err_norm = set_robot_based_on_ee_pos(
            #     self._wrapped_env,
            #     ac,
            #     self.ik_env,
            #     qpos,
            #     qvel,
            #     False,
            #     "cube",
            #     self.config,
            #     self.allowed_collision_pairs,
            #     self.name,
            # )
            mp_to_point(
                self._wrapped_env,
                ac,
                self.ik_env,
                self._wrapped_env.sim.data.qpos.copy(),
                self._wrapped_env.sim.data.qvel.copy(),
                self.allowed_collision_pairs,
                is_grasped=False,
                ignore_object_collision=False,
                planning_time=15,
                get_intermediate_frames=True,
                backtrack_movement_fraction=0.01,
                env_name="SawyerLiftObstacle-v0",
                config=self.config,
                mp_env=self,
            )
            # open gripper and set to state of opened gripper
            self._wrapped_env.sim.data.qpos[
                self._wrapped_env.ref_gripper_joint_pos_indexes
            ] = np.array([-0.0115, -0.0115])
            self._wrapped_env.sim.data.qvel[
                self._wrapped_env.ref_gripper_joint_pos_indexes
            ] = np.array([0.03285798, 0.03401761])
            self._wrapped_env.sim.forward()
            return cube_pos
        elif self.name == "SawyerAssemblyObstacle-v0":
            # teleport close to peg
            if self.use_vision_pose_estimation:
                hole_pos = get_object_pose_from_seg(
                    self._wrapped_env,
                    "4_part4_mesh",
                    "topview",
                    500,
                    500,
                    self._wrapped_env.sim,
                ) + np.array([0.0, -0.3, 0.5])
            else:
                hole_pos = get_site_pose(self._wrapped_env, "hole")[0] + np.array(
                    [0.15, 0.10, 0.3]
                )
            teleport_ac = collections.OrderedDict()
            teleport_ac["default"] = hole_pos  # + np.array([0.00, 0.2, 0.0])
            # rotation so that it can get in between back legs of table
            teleport_ac["quat"] = np.array(
                [-0.69904332, -0.35891423, -0.60671187, 0.12008213]
            )
            # result, err_norm = set_robot_based_on_ee_pos(
            #     self._wrapped_env,
            #     teleport_ac,
            #     self.ik_env,
            #     self._wrapped_env.sim.data.qpos.copy(),
            #     self._wrapped_env.sim.data.qvel.copy(),
            #     False,
            #     "0_part0",
            #     self.config,
            #     self.allowed_collision_pairs,
            #     self.name,
            # )
            mp_to_point(
                self._wrapped_env,
                teleport_ac,
                self.ik_env,
                self._wrapped_env.sim.data.qpos.copy(),
                self._wrapped_env.sim.data.qvel.copy(),
                self.allowed_collision_pairs,
                is_grasped=False,
                ignore_object_collision=False,
                planning_time=15,
                get_intermediate_frames=False,
                backtrack_movement_fraction=0.01,
                env_name="SawyerAssemblyObstacle-v0",
                config=self.config,
                mp_env=self,
            )
            return None
        else:
            if self.use_vision_pose_estimation:
                cube_pos = get_object_pose_from_seg(
                    self._wrapped_env,
                    "cube",
                    "frontview",
                    500,
                    500,
                    self._wrapped_env.sim,
                ) + np.array([-0.1, 0.04, 0.06])
            else:
                right_gripper, left_gripper = (
                    self._wrapped_env.sim.data.get_site_xpos("right_eef"),
                    self._wrapped_env.sim.data.get_site_xpos("left_eef"),
                )
                gripper_site_pos = (right_gripper + left_gripper) / 2.0
                cube_pos = np.array(
                    self._wrapped_env.sim.data.body_xpos[self._wrapped_env.cube_body_id]
                ) + np.array([-0.1, 0.05, 0.06])
            ac = collections.OrderedDict()
            ac["default"] = cube_pos
            # result, err_norm = set_robot_based_on_ee_pos(
            #     self._wrapped_env,
            #     ac,
            #     self.ik_env,
            #     self._wrapped_env.sim.data.qpos.copy(),
            #     self._wrapped_env.sim.data.qvel.copy(),
            #     False,
            #     "cube",
            #     self.config,
            #     self.allowed_collision_pairs,
            #     self.name,
            # )
            mp_to_point(
                self._wrapped_env,
                ac,
                self.ik_env,
                self._wrapped_env.sim.data.qpos.copy(),
                self._wrapped_env.sim.data.qvel.copy(),
                self.allowed_collision_pairs,
                is_grasped=False,
                ignore_object_collision=False,
                planning_time=15,
                get_intermediate_frames=False,
                backtrack_movement_fraction=0.01,
                env_name="SawyerPushObstacle-v0",
                config=self.config,
                mp_env=self,
            )
            return None

    def get_target_pos_no_planner(self):
        if self.name == "SawyerLift-v0":
            return np.array([0.0, 0.0, 0.40])
        elif self.name == "SawyerAssemblyObstacle-v0":
            raise NotImplementedError  # should not be a thing since we just need to place the peg in the hole
        else:
            raise NotImplementedError

    def _check_success(self):
        if self.name == "SawyerLift-v0" or self.name == "SawyerLiftObstacle-v0":
            # copied from sawyer_lift_obstacle.py
            info = {}
            reward = 0

            reach_mult = 0.1
            grasp_mult = 0.35
            lift_mult = 0.5
            hover_mult = 0.7
            cube_body_id = self._wrapped_env.sim.model.body_name2id("cube")

            reward_reach = 0.0
            gripper_site_pos = self._wrapped_env.sim.data.get_site_xpos("grip_site")
            cube_pos = np.array(self._wrapped_env.sim.data.body_xpos[cube_body_id])
            gripper_to_cube = np.linalg.norm(cube_pos - gripper_site_pos)
            reward_reach = (1 - np.tanh(10 * gripper_to_cube)) * reach_mult

            touch_left_finger = False
            touch_right_finger = False
            for i in range(self._wrapped_env.sim.data.ncon):
                c = self._wrapped_env.sim.data.contact[i]
                if c.geom1 == self._wrapped_env.cube_geom_id:
                    if c.geom2 in self._wrapped_env.l_finger_geom_ids:
                        touch_left_finger = True
                    if c.geom2 in self._wrapped_env.r_finger_geom_ids:
                        touch_right_finger = True
                elif c.geom2 == self._wrapped_env.cube_geom_id:
                    if c.geom1 in self._wrapped_env.l_finger_geom_ids:
                        touch_left_finger = True
                    if c.geom1 in self._wrapped_env.r_finger_geom_ids:
                        touch_right_finger = True
            has_grasp = touch_right_finger and touch_left_finger
            reward_grasp = int(has_grasp) * grasp_mult

            reward_lift = 0.0
            object_z_locs = self._wrapped_env.sim.data.body_xpos[cube_body_id][2]
            if reward_grasp > 0.0:
                z_target = self._wrapped_env._get_pos("bin1")[2] + 0.45
                z_dist = np.maximum(z_target - object_z_locs, 0.0)
                reward_lift = grasp_mult + (1 - np.tanh(15 * z_dist)) * (
                    lift_mult - grasp_mult
                )

            reward += max(reward_reach, reward_grasp, reward_lift)
            info = dict(
                reward_reach=reward_reach,
                reward_grasp=reward_grasp,
                reward_lift=reward_lift,
            )
            success = False
            if reward_grasp > 0.0 and np.abs(object_z_locs - z_target) < 0.05:
                reward += 150.0  # this is the success reward
                success = True
            return success
        elif self.name == "SawyerAssemblyObstacle-v0":
            info = {}
            reward = 0
            pegHeadPos = self._wrapped_env.sim.data.get_site_xpos("pegHead")
            hole = self._wrapped_env.sim.data.get_site_xpos("hole")
            dist = np.linalg.norm(pegHeadPos - hole)
            hole_bottom = self._wrapped_env.sim.data.get_site_xpos("hole_bottom")
            dist_to_hole_bottom = np.linalg.norm(pegHeadPos - hole_bottom)
            dist_to_hole = np.linalg.norm(pegHeadPos - hole)
            reward_reach = 0
            if dist < 0.3:
                reward_reach += 0.4 * (1 - np.tanh(15 * dist_to_hole))
            reward += reward_reach
            success = False
            if dist_to_hole_bottom < 0.025:
                success = True
                terminal = True
            return success
        elif self.name == "SawyerPushObstacle-v0":
            self._wrapped_env.compute_reward(None)
            success = self._wrapped_env._success
            return success

    def check_grasp(self):
        return check_grasp(self._wrapped_env, self.name)

    def reset(self):
        # reset step counters
        self.high_level_step = 0
        self.num_steps = 0
        # reset environment
        self._wrapped_env.reset()
        self.ik_env.reset()
        # get reset qpos and qvel
        self.reset_qpos = self._wrapped_env.sim.data.qpos.copy()
        self.reset_qvel = self._wrapped_env.sim.data.qvel.copy()
        # get reset ori
        self.reset_ori = get_site_pose(self._wrapped_env, "grip_site")[
            1
        ]  # in xyzw format
        # self terminal
        self._terminal = False
        # save image of current state
        if (
            not self.plan_to_learned_goals
            and not self.planner_only_actions
            and not self.no_mprl
        ):
            if self.teleport_instead_of_mp:
                pos = self.get_init_target_pos()
                obs = self._get_observation()
                if (
                    self.name == "SawyerLift-v0" or self.name == "SawyerLiftObstacle-v0"
                ):  # open the gripper
                    self._wrapped_env.sim.data.qpos[
                        self._wrapped_env.ref_gripper_joint_pos_indexes
                    ] = np.array([-0.015, -0.015])
                    self._wrapped_env.sim.forward()
            else:
                raise NotImplementedError
        else:
            obs = self._get_observation()
        return obs

    def render(self, **args):
        return (self._wrapped_env.render("rgb_array") * 255).astype(np.uint8)

    def step(self, action):
        is_grasped = False
        if self.plan_to_learned_goals:
            if self.take_planner_step:
                # TODO self.learn_residual
                if self.learn_residual:
                    raise NotImplementedError
                else:
                    delta_pos = action[:3]
                    gripper_ac = np.array([action[-1]])
                    ac = collections.OrderedDict()
                    ac["default"] = delta_pos
                    if (
                        self.name == "SawyerLift-v0"
                        or self.name == "SawyerLiftObstacle-v0"
                    ):
                        ac["gripper"] = gripper_ac
                    if self.planner_command_orientation:
                        delta_rot_mat = euler2mat(action[3:6])
                        target_rot = (
                            delta_rot_mat
                            @ self._wrapped_env.sim.data.get_site_xmat("grip_site")
                        )
                        ac["quat"] = mat2quat(target_rot)
                        ac["quat"] = convert_quat(ac["quat"], to="wxyz")
                    is_grasped = check_grasp(self._wrapped_env, self.name)
                    if self.teleport_instead_of_mp:
                        obj_name = "cube" if self.name == "SawyerLift-v0" else None
                        backtracking_search_from_goal(
                            self._wrapped_env,
                            self.ik_env,
                            ac,
                            is_grasped,
                            self.reset_qpos.copy(),
                            self.reset_qvel.copy(),
                            obj_name,
                            self.config,
                            0.01,
                            self.allowed_collision_pairs,
                            self.name,
                        )
                        self._wrapped_env._episode_length += 1
                        self.ik_env._episode_length += 1
                        r, i = self._wrapped_env.compute_reward(None)
                        d = self._wrapped_env._terminal
                # set stuff to false
                self.take_planner_step = False
                self.high_level_step += 1
            else:
                delta_pos = action[:3]
                gripper_ac = np.array([action[-1]])
                ac = collections.OrderedDict()
                ac["default"] = delta_pos
                if self.name == "SawyerLift-v0" or self.name == "SawyerLiftObstacle-v0":
                    ac["gripper"] = gripper_ac
                if self.planner_command_orientation:
                    delta_rot_mat = euler2mat(action[3:6])
                    target_rot = (
                        delta_rot_mat
                        @ self._wrapped_env.sim.data.get_site_xmat("grip_site")
                    )
                    ac["quat"] = mat2quat(target_rot)
                    ac["quat"] = convert_quat(ac["quat"], to="wxyz")
                converted_ac = cart2joint_ac(
                    self._wrapped_env,
                    self.ik_env,
                    ac,
                    self.reset_qpos,
                    self.reset_qvel,
                    self.config,
                )
                o, r, d, i = self._wrapped_env.step(converted_ac)
                self.current_ll_policy_steps += 1
                if self.current_ll_policy_steps == self.num_ll_actions_per_hl_action:
                    self.take_planner_step = True
                    self.current_ll_policy_steps = 0
                self.num_steps += 1
            self.ep_step_ctr += 1
        else:
            delta_pos = action[:3]
            gripper_ac = np.array([action[-1]])
            ac = collections.OrderedDict()
            ac["default"] = delta_pos
            if self.name == "SawyerLift-v0" or self.name == "SawyerLiftObstacle-v0":
                ac["gripper"] = gripper_ac
            if self.planner_command_orientation:
                delta_rot_mat = euler2mat(action[3:6])
                target_rot = delta_rot_mat @ self._wrapped_env.sim.data.get_site_xmat(
                    "grip_site"
                )
                ac["quat"] = mat2quat(target_rot)
                ac["quat"] = convert_quat(ac["quat"], to="wxyz")
            # always use end effector control
            converted_ac = cart2joint_ac(
                self._wrapped_env,
                self.ik_env,
                ac,
                self.reset_qpos,
                self.reset_qvel,
                self.config,
            )
            o, r, d, i = self._wrapped_env.step(converted_ac)
            # increment steps
            self.num_steps += 1
            self.ep_step_ctr += 1
            # sync wrapped_env steps with steps we are counting
            self._wrapped_env._episode_length = self.ep_step_ctr
            self.ik_env._episode_length = self.ep_step_ctr
            is_grasped = check_grasp(self._wrapped_env, self.name)
            if self.teleport_on_grasp and is_grasped:
                teleport_ac = collections.OrderedDict()
                teleport_ac["default"] = self.get_target_pos_no_planner()
                assert check_grasp(self._wrapped_env, self.name)
                if self.teleport_instead_of_mp:
                    set_robot_based_on_ee_pos(
                        self._wrapped_env,
                        teleport_ac,
                        self.ik_env,
                        self._wrapped_env.sim.data.qpos.copy(),
                        self._wrapped_env.sim.data.qvel.copy(),
                        True,
                        "cube",
                        self.config,
                        self.allowed_collision_pairs,
                    )
                # add reward to count success into action
                r += self._wrapped_env.compute_reward(collections.OrderedDict())[0]
                # also recompute whether done or not
                d = self._wrapped_env._terminal
                # get observation
                o = self._get_observation()
        i["success"] = float(self._wrapped_env._success)  # float(self._check_success())
        i["grasped"] = float(is_grasped)
        if d and self._terminal:
            i["bad_mask"] = np.array([1.0])[:, np.newaxis]
        else:
            i["bad_mask"] = np.array([0.0])[:, np.newaxis]
        i["num_high_level_steps"] = self.high_level_step
        if self.name == "SawyerLift-v0":
            i["distance_to_cube"] = np.linalg.norm(
                get_object_pose(self._wrapped_env, "cube")[:3]
                - get_site_pose(self._wrapped_env, "grip_site")[0]
            )
        self._terminal = d
        i["num_steps"] = self.num_steps
        # delete keys from mopa rl
        if "episode_reward" in i.keys():
            del i["episode_reward"]
        if "episode_success" in i.keys():
            del i["episode_success"]
        if "episode_time" in i.keys():
            del i["episode_time"]
        if "episode_unstable" in i.keys():
            del i["episode_unstable"]
        if "episode_length" in i.keys():
            del i["episode_length"]
        o = self._get_observation()
        assert d == self._wrapped_env._terminal
        assert (
            self._terminal == d
        )  # checks to make sure done is being correctly updated
        # allows for ignore done
        if self.ignore_done:
            d = self.ep_step_ctr == self.horizon
        return o, r, d, i

    def get_image(self, *args, **kwargs):
        # frame = (self._wrapped_env.render("rgb_array") * 255.0).astype(np.uint8)

        frame = cv2.flip(
            self._wrapped_env.sim.render(
                camera_name="frontview",
                width=960,
                height=540,
                depth=False,
            ),
            0,
        )
        return frame
