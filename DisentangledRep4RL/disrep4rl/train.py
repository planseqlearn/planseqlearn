# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time
import datetime
import os
import subprocess
import traceback

from doodad.slurm.slurm_util import wrap_command_with_sbatch_matrix, SlurmConfigMatrix

import dateutil.tz
import sys
import hydra

@hydra.main(config_path="cfgs", config_name="train_config")
def main(cfg):
    import datetime
    import os
    from collections import OrderedDict
    from pathlib import Path


    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

    import hydra
    import numpy as np
    import torch
    import wandb
    from dm_env import specs
    from omegaconf import OmegaConf
    import git
    import ml_runlog
    import dateutil.tz
    import sys
    import json
    from disrep4rl import utils
    from disrep4rl.environments import dmc
    from disrep4rl.environments.adroit_dm_env import make_adroit
    from disrep4rl.environments.distracting_dmc import make_distracting_dmc
    from disrep4rl.environments.kitchen_dm_env import make_kitchen
    from disrep4rl.environments.metaworld_dm_env import make_metaworld
    from disrep4rl.environments.robosuite_dm_env import make_robosuite
    from disrep4rl.logger import Logger, compute_path_info
    from disrep4rl.replay_buffer import ReplayBufferStorage, make_replay_loader
    from disrep4rl.video import TrainVideoRecorder, VideoRecorder
    from rlkit.core import logger as rlkit_logger
    from rlkit.core.eval_util import create_stats_ordered_dict

    torch.backends.cudnn.benchmark = True


    def make_agent(obs_spec, action_spec, agent_cfg, pretrain_cfg):
        assert (
            "pixels" in obs_spec
        ), "Observation spec passed to make_agent must contain a observation named 'pixels'"

        agent_cfg.obs_shape = obs_spec["pixels"].shape
        agent_cfg.action_shape = action_spec.shape
        agent = hydra.utils.instantiate(agent_cfg)
        if "path" in pretrain_cfg:
            agent.load_pretrained_weights(pretrain_cfg.path,
                                        pretrain_cfg.just_encoder_decoders)
        return agent


    def make_env(cfg, is_eval):
        if cfg.task_name.split("_", 1)[0] == "metaworld":
            env = make_metaworld(
                cfg.task_name.split("_", 1)[1], cfg.frame_stack, cfg.action_repeat,
                cfg.discount, cfg.seed, cfg.camera_name,
                cfg.add_segmentation_to_obs, cfg.noisy_mask_drop_prob, cfg.use_rgbm,
                cfg.slim_mask_cfg, cfg.mprl)
        elif cfg.task_name.split("_", 1)[0] == "robosuite":
            env = make_robosuite(
                cfg.task_name.split("_", 1)[1], cfg.frame_stack, cfg.action_repeat,
                cfg.discount, cfg.seed, cfg.camera_name,
                cfg.add_segmentation_to_obs, cfg.noisy_mask_drop_prob, cfg.use_rgbm,
                cfg.slim_mask_cfg, cfg.mprl, cfg.path_length, cfg.vertical_displacement,
                cfg.hardcoded_orientations, cfg.valid_obj_names, cfg.steps_of_high_level_plan_to_complete, cfg.use_proprio,
                cfg.use_fixed_plus_wrist_view, cfg.pose_sigma, cfg.noisy_pose_estimates, cfg.hardcoded_high_level_plan)
        elif cfg.task_name.split("_", 1)[0] == "adroit":
            env = make_adroit(
                cfg.task_name.split("_", 1)[1], cfg.frame_stack, cfg.action_repeat,
                cfg.discount, cfg.seed, cfg.camera_name,
                cfg.add_segmentation_to_obs, cfg.noisy_mask_drop_prob, cfg.use_rgbm,
                cfg.slim_mask_cfg)
        elif cfg.task_name.split("_", 1)[0] == "kitchen":
            env = make_kitchen(
                cfg.task_name.split("_", 1)[1], cfg.frame_stack, cfg.action_repeat,
                cfg.discount, cfg.seed, cfg.camera_name,
                cfg.add_segmentation_to_obs, cfg.noisy_mask_drop_prob, cfg.use_rgbm,
                cfg.slim_mask_cfg, cfg.path_length, cfg.mprl)
        elif cfg.task_name.split("_", 1)[0] == "distracting":
            background_dataset_videos = "val" if is_eval else "train"
            env = make_distracting_dmc(
                cfg.task_name.split("_", 1)[1], cfg.frame_stack, cfg.action_repeat,
                cfg.seed, cfg.add_segmentation_to_obs, cfg.distraction.difficulty,
                cfg.distraction.types, cfg.distraction.dataset_path,
                background_dataset_videos, cfg.noisy_mask_drop_prob, cfg.use_rgbm,
                cfg.slim_mask_cfg)
        else:
            env = dmc.make(cfg.task_name, cfg.frame_stack, cfg.action_repeat,
                        cfg.seed)
        return env


    class Workspace:

        def __init__(self, cfg):
            self.work_dir = Path.cwd()
            print(f"workspace: {self.work_dir}")
            rlkit_logger.set_snapshot_dir(self.work_dir)
            rlkit_logger.use_wandb = False
            # create progress csv file 
            rlkit_logger.add_tabular_output(os.path.join(self.work_dir, "progress.csv"))

            self.cfg = cfg
            utils.set_seed_everywhere(cfg.seed)
            self.device = torch.device(cfg.device)

            self.logger = Logger(self.work_dir,
                                use_tb=self.cfg.use_tb,
                                use_wandb=self.cfg.use_wandb)

            self.train_env = make_env(self.cfg, is_eval=False)
            self.eval_env = make_env(self.cfg, is_eval=True)
            if self.cfg.has_success_metric:
                reward_spec = OrderedDict([
                    ("reward", specs.Array((1,), np.float32, "reward")),
                    ("success", specs.Array((1,), np.int16, "reward"))
                ])
            else:
                reward_spec = specs.Array((1,), np.float32, "reward")

            discount_spec = specs.Array((1,), np.float32, "discount")
            data_specs = {
                "observation": self.train_env.observation_spec(),
                "action": self.train_env.action_spec(),
                "reward": reward_spec,
                "discount": discount_spec
            }
            self.replay_storage = ReplayBufferStorage(data_specs,
                                                    self.work_dir / "buffer")

            self.replay_loader = make_replay_loader(
                self.work_dir / "buffer", self.cfg.replay_buffer_size,
                self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
                self.cfg.save_buffer_snapshot, self.cfg.nstep, self.cfg.discount,
                self.cfg.has_success_metric)
            self._replay_iter = None

            self.agent = make_agent(self.train_env.observation_spec(),
                                    self.train_env.action_spec(), self.cfg.agent,
                                    self.cfg.pretrain)

            self.video_recorder = VideoRecorder(
                self.work_dir if self.cfg.save_video else None,
                metaworld_camera_name=self.cfg.camera_name if cfg.task_name.split("_", 1)[0] == "metaworld" else None,
                use_wandb=self.cfg.use_wandb)
            self.train_video_recorder = TrainVideoRecorder(
                self.work_dir if self.cfg.save_train_video else None)

            cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)
            with open(os.path.join(self.work_dir, "variant.json"), "w") as f:
                json.dump(cfg_dict, f, indent=4)
            if self.cfg.use_wandb:
                wandb_worked = False
                while not wandb_worked:
                    try:
                        self.wandb_run = wandb.init(project=self.cfg.wandb.project_name,
                                                    config=cfg_dict,
                                                    name=self.cfg.wandb.run_name)
                        wandb_worked = True
                    except:
                        print(traceback.format_exc())
                        print("Wandb failed, retrying...")
                        time.sleep(5)
                
                if not self.cfg.debug:
                    print("Updating run log...")
                    import rlkit
                    import disrep4rl
                    rlkit_dir = os.path.dirname(rlkit.__file__)
                    disrep4rl_dir = os.path.dirname(disrep4rl.__file__)
                    directory = os.path.dirname(disrep4rl_dir)
                    repo = git.Repo(directory)
                    branch_name = repo.active_branch.name
                    ml_runlog.init(
                        os.path.join(
                            rlkit_dir, "launchers", "service_account_credentials.json"
                        ),
                        self.cfg.sheet_name,
                    )

                    # The first argument in the sys.argv list is the name of the script itself
                    script_name = sys.argv[0]
                    # The rest of the arguments are the command-line arguments used to launch the script
                    command_line_args = sys.argv[1:]
                    # Join the command-line arguments into a string
                    command = " ".join(("python", script_name, "", *command_line_args))

                    ml_runlog.log_data(
                        datetime=datetime.datetime.now(dateutil.tz.tzlocal()).isoformat(),
                        exp_prefix=self.cfg.wandb.run_name,
                        run_name=self.cfg.wandb.run_name,
                        run_url=self.wandb_run.get_url(),
                        machine=os.uname().nodename,
                        log_dir=self.work_dir,
                        branch=branch_name,
                        commit_hash=repo.head.commit.hexsha,
                        cmdline_args=command,
                        comments="",
                    )

                    

            self.timer = utils.Timer()
            self._global_step = 0
            self._global_episode = 0
            self._max_success_rate = 0

        @property
        def global_step(self):
            return self._global_step

        @property
        def global_episode(self):
            return self._global_episode

        @property
        def global_frame(self):
            return self.global_step * self.cfg.action_repeat

        @property
        def replay_iter(self):
            if self._replay_iter is None:
                self._replay_iter = iter(self.replay_loader)
            return self._replay_iter

        def eval(self, best_eval_score):
            step, episode, total_reward = 0, 0, 0
            if self.cfg.has_success_metric:
                mean_max_success, mean_mean_success, mean_last_success = 0, 0, 0
            eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
            self.video_recorder.init(self.eval_env, enabled=True)

            all_infos = [] # list of all reward dictionaries from all episodes
            # remember to also log all rewards as well as time per episode 
            all_rewards = []
            all_times = []
            all_path_lengths = []
            episode_reward = 0
            num_successes = 0

            while eval_until_episode(episode):
                if self.cfg.has_success_metric:
                    current_episode_max_success = 0
                    current_episode_mean_success = 0
                    current_episode_last_success = 0
                current_episode_step = 0
                time_step = self.eval_env.reset()
                self.video_recorder.record(self.eval_env)
                ep_infos = [] # initialize list of reward dictionaries from current episode
                start_time = time.time()

                while not time_step.last():
                    current_episode_step += 1
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        action = self.agent.act(time_step.observation,
                                                self.global_step,
                                                eval_mode=True)
                    time_step = self.eval_env.step(action)
                    self.video_recorder.record(self.eval_env)
                    if self.cfg.has_success_metric:
                        total_reward += time_step.reward["reward"]
                        episode_reward += time_step.reward["reward"]
                        success = int(time_step.reward["success"])
                        current_episode_max_success = max(
                            current_episode_max_success, success)
                        current_episode_last_success = success
                        current_episode_mean_success += success
                    else:
                        total_reward += time_step.reward
                    ep_infos.append(time_step.reward)
                    step += 1
                all_infos.append(ep_infos)
                all_path_lengths.append(current_episode_step)
                if self.cfg.has_success_metric:
                    mean_max_success += current_episode_max_success
                    mean_last_success += current_episode_last_success
                    mean_mean_success += current_episode_mean_success / current_episode_step
                    if current_episode_max_success > 0:
                        num_successes += 1
                episode += 1
                all_rewards.append(episode_reward)
                episode_reward = 0
                all_times.append(time.time() - start_time)
            self.video_recorder.save(f"{self.global_frame}",
                                        step=self.global_frame)
            
            # compute info for rlkit logging 
            statistics = compute_path_info(all_infos)
            # update with total reward and time  
            statistics.update(
                create_stats_ordered_dict(
                    "Returns",
                    all_rewards,
                    exclude_max_min=False,
                )
            )
            statistics.update(
                create_stats_ordered_dict(
                    "Path Length",
                    all_path_lengths,
                    exclude_max_min=False,
                )
            )
            statistics.update(
                create_stats_ordered_dict(
                    "Time per episode (s)",
                    all_times,
                    exclude_max_min=True,
                )
            )
            rlkit_logger.record_dict(statistics, prefix="Evaluation/")
            # compute max success rate and log
            self._max_success_rate = max(self._max_success_rate, num_successes / episode)
            rlkit_logger.record_dict({"Max overall success rate": self._max_success_rate}, prefix="Evaluation/")

            with self.logger.log_and_dump_ctx(self.global_frame, ty="eval") as log:
                log("episode_reward", total_reward / episode)
                log("episode_length", step * self.cfg.action_repeat / episode)
                log("episode", self.global_episode)
                log("step", self.global_step)
                if self.cfg.has_success_metric:
                    log("max_success", mean_max_success / episode)
                    log("last_success", mean_last_success / episode)
                    log("mean_success", mean_mean_success / episode)
            if self.cfg.has_success_metric:
                episode_score = mean_max_success / episode
            else:
                episode_score = total_reward/episode
            # try to save snapshot
            if self.cfg.save_snapshot:
                if episode_score > best_eval_score:
                    self.save_snapshot(file_name="best_eval.pt")
            best_eval_score = max(best_eval_score, episode_score)

        def train(self):
            # predicates
            train_until_step = utils.Until(self.cfg.num_train_frames,
                                        self.cfg.action_repeat)
            seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                        self.cfg.action_repeat)
            eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                        self.cfg.action_repeat)

            episode_step, episode_reward = 0, 0
            if self.cfg.has_success_metric:
                mean_success = 0
                max_success = 0
                last_success = 0

            # Score is mean success if task has success metric, else it is episode reward
            best_episode_score = -np.inf
            best_eval_score = -np.inf

            time_step = self.train_env.reset()
            self.replay_storage.add(time_step)
            self.train_video_recorder.init(time_step.observation)
            metrics = None
            epoch = 0

             # rlkit logging information
            all_infos = []
            all_rewards = []
            all_times = []
            all_path_lengths = []
            ep_infos = []
            start = time.time()
            total_start_time = time.time()
            epoch_start = time.time()
            epoch_train_calls = 0
            epoch_steps_total = 0

            while train_until_step(self.global_step):
                if time_step.last():
                    all_times.append(time.time() - start)

                    self._global_episode += 1
                    self.train_video_recorder.save(f"{self.global_frame}.mp4")
                    # wait until all the metrics schema is populated
                    if metrics is not None:
                        # log stats
                        elapsed_time, total_time = self.timer.reset()
                        episode_frame = episode_step * self.cfg.action_repeat
                        with self.logger.log_and_dump_ctx(self.global_frame,
                                                        ty="train") as log:
                            log("fps", episode_frame / elapsed_time)
                            log("total_time", total_time)
                            log("episode_reward", episode_reward)
                            log("episode_length", episode_frame)
                            log("episode", self.global_episode)
                            log("buffer_size", len(self.replay_storage))
                            log("step", self.global_step)
                            if self.cfg.has_success_metric:
                                log("mean_success", mean_success / episode_step)
                                log("max_success", max_success)
                                log("last_success", last_success)

                    # reset env
                    time_step = self.train_env.reset()
                    self.replay_storage.add(time_step)
                    self.train_video_recorder.init(time_step.observation)

                    if self.cfg.has_success_metric:
                        episode_score = mean_success / episode_step
                    else:
                        episode_score = episode_reward

                    # try to save snapshot
                    if self.cfg.save_snapshot:
                        self.save_snapshot(file_name="latest.pt")

                        if episode_score > best_episode_score:
                            self.save_snapshot(file_name="best.pt")

                    best_episode_score = max(episode_score, best_episode_score)

                    all_path_lengths.append(episode_step)
                    all_rewards.append(episode_reward)
                    episode_step = 0
                    episode_reward = 0
                    all_infos.append(ep_infos)
                    ep_infos = [] # set ep_infos to empty list and append to end of all_infos
                    if self.cfg.has_success_metric:
                        mean_success = 0
                        max_success = 0
                        last_success = 0

                # try to evaluate
                if eval_every_step(self.global_step) and self.global_step != 0: # not evaluating at step 0 for logging
                    self.logger.log("eval_total_time", self.timer.total_time(),
                                    self.global_frame)
                    eval_time = time.time()
                    self.eval(best_eval_score)
                    eval_time = time.time() - eval_time
                    epoch_time = time.time() - epoch_start
                    # compute info for rlkit logging 
                    statistics = compute_path_info(all_infos)
                    # update with total reward and time  
                    statistics.update(
                        create_stats_ordered_dict(
                            "Returns",
                            all_rewards,
                            exclude_max_min=False,
                        )
                    )
                    statistics.update(
                        create_stats_ordered_dict(
                            "Path Length",
                            all_path_lengths,
                            exclude_max_min=False,
                        )
                    )
                    statistics.update(
                        create_stats_ordered_dict(
                            "Time per episode (s)",
                            all_times,
                            exclude_max_min=True,
                        )
                    )
                    statistics.update(
                        {"Num steps total": epoch_steps_total}
                    )
                    rlkit_logger.record_dict(statistics, prefix="Exploration/")
                    rlkit_logger.record_dict({"Epoch": epoch}, prefix="")
                    rlkit_logger.record_dict({"Time for epoch (s)": epoch_time}, prefix="Time/")
                    rlkit_logger.record_dict({"Training and Exploration Time (s)": epoch_time - eval_time}, prefix="Time/")
                    rlkit_logger.record_dict({"Num train calls": epoch_train_calls}, prefix="Trainer/")
                    rlkit_logger.record_dict({"Total time (s)": time.time() - total_start_time}, prefix="Time/")
                    rlkit_logger.record_dict({"Total train episodes": self._global_episode}, prefix="Exploration")
                    rlkit_logger.dump_tabular(with_prefix=False, with_timestamp=False)
                    # reset information 
                    all_infos = []
                    all_times = []
                    all_rewards = []
                    all_path_lengths = []
                    epoch += 1
                    epoch_steps_total = 0
                    epoch_train_calls = 0
                    epoch_start = time.time()

                # sample action
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=False)

                # try to update the agent
                if not seed_until_step(self.global_step):
                    metrics = self.agent.update(self.replay_iter, self.global_step)
                    self.logger.log_metrics(metrics, self.global_frame, ty="train")
                    epoch_train_calls += 1

                if episode_step == 0: # accurate timekeeping per episode 
                    start = time.time() 
                # take env step
                time_step = self.train_env.step(action)
                if self.cfg.has_success_metric:
                    episode_reward += time_step.reward["reward"]
                    success = int(time_step.reward["success"])
                    max_success = max(max_success, success)
                    last_success = success
                    mean_success += success
                    ep_infos.append(time_step.reward)
                else:
                    episode_reward += time_step.reward
                self.replay_storage.add(time_step)
                self.train_video_recorder.record(time_step.observation)
                episode_step += 1
                epoch_steps_total += 1
                self._global_step += 1

        def save_snapshot(self, file_name="snapshot.pt"):
            snapshot = self.work_dir / file_name
            keys_to_save = ["agent", "timer", "_global_step", "_global_episode"]
            payload = {k: self.__dict__[k] for k in keys_to_save}
            cfg_keys_to_save = [
                "has_success_metric", "task_name", "frame_stack", "action_repeat",
                "discount", "add_segmentation_to_obs"
            ]
            if "camera_name" in self.cfg:
                cfg_keys_to_save.append("camera_name")
            payload.update({k: self.cfg[k] for k in cfg_keys_to_save})

            with snapshot.open("wb") as f:
                torch.save(payload, f)

        def load_snapshot(self):
            snapshot = self.work_dir / "snapshot.pt"
            with snapshot.open("rb") as f:
                payload = torch.load(f)
            for k, v in payload.items():
                self.__dict__[k] = v
        
    root_dir = Path.cwd()
    workspace = Workspace(cfg)
    snapshot = root_dir / "snapshot.pt"
    if snapshot.exists():
        print(f"resuming: {snapshot}")
        workspace.load_snapshot()
    workspace.train()


if __name__ == "__main__":
    # This block of code takes the hydra command and converts it to a command to run on the matrix if specified
    SINGULARITY_PRE_CMDS = [
        "export MUJOCO_GL='egl'",
        "export MKL_THREADING_LAYER=GNU",
        "export D4RL_SUPPRESS_IMPORT_ERROR='1'",
    ]
    slurm_config = SlurmConfigMatrix(
        partition="russ_reserved",
        time="72:00:00",
        n_gpus=1,
        n_cpus_per_gpu=20,
        mem="62g",
        extra_flags="--exclude=matrix-1-[4,6,8,10,12,16,18,20],matrix-0-[38]", #throw out non-RTX
    )
    exp_id = "Default_Experiment_ID"
    experiment_subdir = None
    matrix = False
    command_line_args = []
    for pair in sys.argv[1:]:
        if pair.startswith('experiment_id='):
            exp_id = pair.split('=')[1]
        if pair.startswith('experiment_subdir='):
            experiment_subdir = pair.split('=')[1]
        if pair == 'matrix=True':
            matrix = True
        else:
            command_line_args.append(pair)

    # get log directory
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime(f"%Y_%m_%d_%H_%M_%S_{now.microsecond}")
    cwd = os.getcwd()
    if experiment_subdir is not None:
        log_dir = os.path.join(cwd, f"exp_local/{experiment_subdir}/%s_%s" % (exp_id, timestamp))
    else:
        log_dir = os.path.join(cwd, "exp_local/%s_%s" % (exp_id, timestamp))
    os.makedirs(log_dir)
    sys.argv.append(f'hydra.run.dir={log_dir}')
    # get python command: 
    script_name = sys.argv[0]
    # Join the command-line arguments into a string
    python_cmd = subprocess.check_output("which python", shell=True).decode(
                "utf-8"
            )[:-1]
    command = " ".join((python_cmd, script_name, "", *command_line_args))
    logdir = log_dir
    singularity_pre_cmds = " && ".join(SINGULARITY_PRE_CMDS)
    slurm_cmd = wrap_command_with_sbatch_matrix(
        "/opt/singularity/bin/singularity exec --nv /projects/rsalakhugroup/containers/mprl.sif /bin/zsh -c \"" + singularity_pre_cmds + ' && source ~/.zshrc && conda activate drqv2 && ' + command + "\"",
        slurm_config,
        logdir,
    )
    if matrix:
        print(slurm_cmd)
        os.system(slurm_cmd)
    else:
        main()
