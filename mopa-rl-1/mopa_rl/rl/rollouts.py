from collections import defaultdict, OrderedDict

import numpy as np
import torch
import cv2
import gym

from mopa_rl.env.inverse_kinematics import qpos_from_site_pose_sampling, qpos_from_site_pose
from mopa_rl.util.logger import logger
from mopa_rl.util.env import joint_convert, mat2quat, quat_mul, rotation_matrix, quat2mat
from mopa_rl.util.gym import action_size
from mopa_rl.util.info import Info


class Rollout(object):
    def __init__(self):
        self._history = defaultdict(list)

    def add(self, data):
        for key, value in data.items():
            self._history[key].append(value)

    def __len__(self):
        return len(self._history["ob"])

    def get(self):
        batch = {}
        batch["ob"] = self._history["ob"]
        batch["ac"] = self._history["ac"]
        batch["meta_ac"] = self._history["meta_ac"]
        batch["ac_before_activation"] = self._history["ac_before_activation"]
        batch["done"] = self._history["done"]
        batch["rew"] = self._history["rew"]
        batch["intra_steps"] = self._history["intra_steps"]
        self._history = defaultdict(list)
        return batch


class RolloutRunner(object):
    def __init__(self, config, env, env_eval, pi):
        self._config = config
        self._env = env
        self._env_eval = env_eval
        self._pi = pi
        #self._ik_env = gym.make(config.env, **config.__dict__)
        # set ik env to None now since none of our experiments require it
        self._ik_env = None

    def run(
        self,
        max_step=10000,
        is_train=True,
        random_exploration=False,
        every_steps=None,
        every_episodes=None,
    ):
        """
        Collects trajectories and yield every @every_steps/@every_episodes.
        Args:
            is_train: whether rollout is for training or evaluation.
            every_steps: if not None, returns rollouts @every_steps
            every_episodes: if not None, returns rollouts @every_epiosdes
        """
        if every_steps is None and every_episodes is None:
            raise ValueError("Both every_steps and every_episodes cannot be None")
        config = self._config
        device = config.device
        env = self._env if is_train else self._env_eval
        ik_env = self._ik_env if config.use_ik_target else None
        pi = self._pi

        rollout = Rollout()
        reward_info = Info()
        ep_info = Info()

        step = 0
        episode = 0

        while True:
            done = False
            ep_len = 0
            ep_rew = 0
            ob = env.reset()
            # modification for robosuite lift
            env._episode_length = 0
            env.epsiode_reward = 0
            if config.use_ik_target:
                ik_env.reset()

            # modify observation to fit their format
            if config.env == "Lift":
                observation = ob.copy()
                ob = OrderedDict()
                ob['default'] = observation 

            # run rollout
            meta_ac = None
            while not done and ep_len < max_step:
                env_step = 0
                ll_ob = ob.copy()
                ac, ac_before_activation, stds = pi.act(
                    ll_ob,
                    is_train=is_train,
                    return_stds=True,
                    random_exploration=random_exploration,
                )
                rollout.add(
                    {
                        "ob": ll_ob,
                        "meta_ac": meta_ac,
                        "ac": ac,
                        "ac_before_activation": ac_before_activation,
                    }
                )
                if config.use_ik_target:  # IK
                    converted_ac = self._cart2joint_ac(env, ik_env, ac)
                    ob, reward, done, info = env.step(converted_ac)
                    rollout.add({"done": done, "rew": reward})
                    ep_len += 1
                    step += 1
                    ep_rew += reward
                    env_step += 1
                    reward_info.add(info)
                else:
                    if config.expand_ac_space:  # large action space
                        diff = (
                            ac["default"][env.ref_joint_pos_indexes]
                            * config.action_range
                        )
                        actions = pi.interpolate_ac(ac, env._ac_scale, diff)
                        intra_steps = 0
                        meta_rew = 0
                        for j, inter_ac in enumerate(actions):
                            ob, reward, done, info = env.step(inter_ac)
                            ep_len += 1
                            step += 1
                            env_step += 1
                            ep_rew += reward
                            meta_rew += reward * config.discount_factor ** j
                            reward_info.add(info)
                            if done or ep_len >= max_step:
                                break
                        rollout.add(
                            {"done": done, "rew": meta_rew, "intra_steps": intra_steps}
                        )
                    else:
                        if config.env != "Lift":
                            ob, reward, done, info = env.step(ac)
                        else:
                            observation, reward, done, info = env.step(ac['default'])
                            ob = OrderedDict()
                            ob['default'] = observation.copy()
                            info["episode_success"] = int(env.env._check_success())
                            if info["episode_success"]:
                                reward += 150.0
                            info["reward"] = reward
                            done = ep_len == env.horizon - 1
                        rollout.add({"done": done, "rew": reward})
                        ep_len += 1
                        step += 1
                        env_step += 1
                        ep_rew += reward
                        if config.env == "Lift":
                            env._episode_length += 1
                            env._episode_reward += reward
                        reward_info.add(info)
                if every_steps is not None and step % every_steps == 0:
                    # last frame
                    ll_ob = ob.copy()
                    rollout.add({"ob": ll_ob, "meta_ac": meta_ac})
                    ep_info.add({"env_step": env_step})
                    yield rollout.get(), ep_info.get_dict(only_scalar=True)
            ep_info.add({"len": ep_len, "rew": ep_rew})
            reward_info_dict = reward_info.get_dict(reduction="sum", only_scalar=True)
            ep_info.add(reward_info_dict)
            logger.info(
                "Ep %d rollout: %s",
                episode,
                {
                    k: v
                    for k, v in reward_info_dict.items()
                    if not "qpos" in k and np.isscalar(v)
                },
            )

            episode += 1

    def run_episode(
        self, max_step=10000, is_train=True, record=False, random_exploration=False
    ):
        config = self._config
        device = config.device
        env = self._env if is_train else self._env_eval
        ik_env = self._ik_env if config.use_ik_target else None
        pi = self._pi

        rollout = Rollout()
        reward_info = Info()
        ep_info = Info()

        done = False
        ep_len = 0
        ep_rew = 0
        ob = env.reset()
        # modify observation to fit their format
        if config.env == "Lift":
            observation = ob.copy()
            ob = OrderedDict()
            ob['default'] = observation 
        if config.use_ik_target:
            ik_env.reset()
        self._record_frames = []
        if record:
            self._store_frame(env)

        # buffer to save qpos
        saved_qpos = []

        # run rollout
        meta_ac = None
        total_contact_force = 0.0
        while not done and ep_len < max_step:

            ll_ob = ob.copy()
            ac, ac_before_activation, stds = pi.act(
                ll_ob,
                is_train=is_train,
                return_stds=True,
                random_exploration=random_exploration,
            )

            rollout.add(
                {
                    "ob": ll_ob,
                    "meta_ac": meta_ac,
                    "ac": ac,
                    "ac_before_activation": ac_before_activation,
                }
            )
            if config.use_ik_target:
                converted_ac = self._cart2joint_ac(env, ik_env, ac)
                ob, reward, done, info = env.step(converted_ac)
                rollout.add({"done": done, "rew": reward})
                ep_len += 1
                ep_rew += reward
                reward_info.add(info)
            else:
                if config.expand_ac_space:
                    diff = (
                        ac["default"][env.ref_joint_pos_indexes] * config.action_range
                    )
                    actions = pi.interpolate_ac(ac, env._ac_scale, diff)
                    intra_steps = 0
                    meta_rew = 0
                    for j, inter_ac in enumerate(actions):
                        ob, reward, done, info = env.step(inter_ac)
                        ep_len += 1
                        ep_rew += reward
                        meta_rew += reward * config.discount_factor ** j
                        reward_info.add(info)
                        contact_force = env.get_contact_force()
                        total_contact_force += contact_force
                        if record:
                            frame_info = info.copy()
                            frame_info["ac"] = ac["default"]
                            frame_info["contact_force"] = contact_force
                            if config.use_ik_target:
                                frame_info["converted_ac"] = converted_ac["default"]
                            frame_info["std"] = np.array(
                                stds["default"].detach().cpu()
                            )[0]
                            self._store_frame(env, frame_info)
                        if done or ep_len >= max_step:
                            break
                    rollout.add(
                        {"done": done, "rew": meta_rew, "intra_steps": intra_steps}
                    )
                else:
                    if config.env != "Lift":
                            ob, reward, done, info = env.step(ac)
                    else:
                        observation, reward, done, info = env.step(ac['default'])
                        ob = OrderedDict()
                        ob['default'] = observation.copy()
                        info["episode_success"] = int(env.env._check_success())
                        if info["episode_success"]:
                            reward += 150
                        info["reward"] = reward
                        done = ep_len == env.horizon - 1
                    rollout.add({"done": done, "rew": reward})
                    ep_len += 1
                    ep_rew += reward
                    if config.env == "Lift":
                        env._episode_length += 1
                        env._episode_reward += reward
                    reward_info.add(info)

                #contact_force = env.get_contact_force()
                #total_contact_force += contact_force

                if record and not config.expand_ac_space:
                    frame_info = info.copy()
                    frame_info["ac"] = ac["default"]
                    #frame_info["contact_force"] = contact_force
                    if config.use_ik_target:
                        frame_info["converted_ac"] = converted_ac["default"]
                    frame_info["std"] = np.array(stds["default"].detach().cpu())[0]

                    self._store_frame(env, frame_info)

        # last frame
        ll_ob = ob.copy()
        rollout.add({"ob": ll_ob, "meta_ac": meta_ac})

        ep_info.add(
            {
                "len": ep_len,
                "rew": ep_rew,
                "avg_conntact_force": total_contact_force / ep_len,
            }
        )
        reward_info_dict = reward_info.get_dict(reduction="sum", only_scalar=True)
        ep_info.add(reward_info_dict)
        return rollout.get(), ep_info.get_dict(only_scalar=True), self._record_frames

    def _cart2joint_ac(self, env, ik_env, ac):
        config = self._config
        curr_qpos = env.sim.data.qpos.copy()
        target_cart = np.clip(
            env.sim.data.get_site_xpos(config.ik_target)[: len(env.min_world_size)]
            + config.action_range * ac["default"],
            env.min_world_size,
            env.max_world_size,
        )
        if len(env.min_world_size) == 2:
            target_cart = np.concatenate(
                (
                    target_cart,
                    np.array([env.sim.data.get_site_xpos(config.ik_target)[2]]),
                )
            )

        if "quat" in ac.keys():
            target_quat = mat2quat(env.sim.data.get_site_xmat(config.ik_target))
            target_quat = target_quat[[3, 0, 1, 1]]
            target_quat = quat_mul(
                target_quat,
                (ac["quat"] / np.linalg.norm(ac["quat"])).astype(np.float64),
            )
        else:
            target_quat = None
        ik_env.set_state(curr_qpos.copy(), env.data.qvel.copy())
        result = qpos_from_site_pose(
            ik_env,
            config.ik_target,
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
            target_qpos[env.ref_joint_pos_indexes]
            - curr_qpos[env.ref_joint_pos_indexes]
        ) / env._ac_scale
        if "gripper" in ac.keys():
            pre_converted_ac = np.concatenate((pre_converted_ac, ac["gripper"]))
        converted_ac = OrderedDict([("default", pre_converted_ac)])
        return converted_ac

    def _store_frame(self, env, info={}):
        color = (200, 200, 200)

        text = "{:4} {}".format(env._episode_length, env._episode_reward)

        if self._config.env != "Lift":
            frame = env.render("rgb_array") * 255.0
        else:
            frame = np.flipud(env._get_observations()["frontview_image"])

        if self._config.vis_info:
            fheight, fwidth = frame.shape[:2]
            frame = np.concatenate([frame, np.zeros((fheight, fwidth, 3))], 0)

            if self._config.record_caption:
                font_size = 0.4
                thickness = 1
                offset = 12
                x, y = 5, fheight + 10
                cv2.putText(
                    frame,
                    text,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_size,
                    (255, 255, 0),
                    thickness,
                    cv2.LINE_AA,
                )
                for i, k in enumerate(info.keys()):
                    v = info[k]
                    key_text = "{}: ".format(k)
                    (key_width, _), _ = cv2.getTextSize(
                        key_text, cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness
                    )

                    cv2.putText(
                        frame,
                        key_text,
                        (x, y + offset * (i + 2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_size,
                        (66, 133, 244),
                        thickness,
                        cv2.LINE_AA,
                    )

                    cv2.putText(
                        frame,
                        str(v),
                        (x + key_width, y + offset * (i + 2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_size,
                        (255, 255, 255),
                        thickness,
                        cv2.LINE_AA,
                    )

        self._record_frames.append(frame)
