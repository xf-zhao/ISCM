import warnings
import wandb

warnings.filterwarnings("ignore", category=DeprecationWarning)

from tqdm import tqdm
import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"
os.environ["MUJOCO_GL"] = "glfw"

from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs

import dmc, td, present
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import (
    TrainVideoRecorder,
    VideoRecorder,
    TrainVideoRecorderPlus,
    VideoRecorderPlus,
)

torch.backends.cudnn.benchmark = True

VIDEO_RECORDING_TYPES = [
    "pixels",
    "pixeljoints",
    "spectrogram_simple",
    "spectrogram_fusion",
    "spectrogram_noisy",
]


def make_agent(obs_type, obs_spec, action_spec, num_expl_steps, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        if cfg.device_id < 0:
            cfg.device_id = (cfg.seed + 3) % 4
        DISPLAY = f":80.{cfg.device_id}"
        os.environ["DISPLAY"] = DISPLAY
        print(f"Using DISPLAY={DISPLAY}.")
        if cfg.device == "cuda":
            torch.cuda.set_device(cfg.device_id)
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # create logger
        if cfg.use_wandb:
            exp_name = "_".join(
                [cfg.experiment, cfg.agent.name, cfg.task, cfg.obs_type, str(cfg.seed),]
            )
            wandb.init(
                project=cfg.project, group=cfg.agent.name, name=exp_name, config=cfg
            )
        # create logger
        self.logger = Logger(self.work_dir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb)
        # create envs

        if cfg.task.startswith("manipulation"):
            self.train_env = td.make(
                cfg.task,
                cfg.obs_type,
                cfg.frame_stack,
                cfg.seed,
                screen_width=cfg.screen_width,
                screen_height=cfg.screen_height,
                port=cfg.port,
                fake_audio=True,
            )
            self.eval_env = td.make(
                cfg.task,
                cfg.obs_type,
                cfg.frame_stack,
                cfg.seed,
                screen_width=cfg.screen_width,
                screen_height=cfg.screen_height,
                port=cfg.port - 1,
                fake_audio=True,
            )
        elif cfg.task.startswith("presentation"):
            self.train_env = present.make(
                cfg.task, cfg.obs_type, cfg.frame_stack, cfg.seed
            )
            self.eval_env = present.make(
                cfg.task, cfg.obs_type, cfg.frame_stack, cfg.seed
            )
        else:
            self.train_env = dmc.make(
                cfg.task, cfg.obs_type, cfg.frame_stack, cfg.action_repeat, cfg.seed
            )
            self.eval_env = dmc.make(
                cfg.task, cfg.obs_type, cfg.frame_stack, cfg.action_repeat, cfg.seed
            )

        # create agent
        self.agent = make_agent(
            cfg.obs_type,
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            cfg.num_seed_frames // cfg.action_repeat,
            cfg.agent,
        )

        # initialize from pretrained
        if cfg.snapshot_ts > 0:
            pretrained_agent = self.load_snapshot()["agent"]
            print("Initilizing agent..")
            self.agent.init_from(pretrained_agent, cfg.snapshot_rerandom)

        # get meta specs
        meta_specs = self.agent.get_meta_specs()
        # create replay buffer
        data_specs = (
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            specs.Array((1,), np.float32, "reward"),
            specs.Array((1,), np.float32, "discount"),
        )

        # create data storage
        self.replay_storage = ReplayBufferStorage(
            data_specs, meta_specs, self.work_dir / "buffer"
        )

        # create replay buffer
        self.replay_loader = make_replay_loader(
            self.replay_storage,
            cfg.replay_buffer_size,
            cfg.batch_size,
            cfg.replay_buffer_num_workers,
            False,
            cfg.nstep,
            cfg.discount,
        )
        self._replay_iter = None

        # create video recorders
        if cfg.obs_type in (
            "spectrogram_simple",
            "spectrogram_fusion",
            "spectrogram_noisy",
        ):
            _video_recorder = VideoRecorderPlus
            _train_video_recorder = TrainVideoRecorderPlus
        else:
            _video_recorder = VideoRecorder
            _train_video_recorder = TrainVideoRecorder
        self.video_recorder = _video_recorder(
            self.work_dir if cfg.save_video else None, use_wandb=self.cfg.use_wandb,
        )
        self.train_video_recorder = _train_video_recorder(
            self.work_dir if cfg.save_train_video else None,
            use_wandb=self.cfg.use_wandb,
        )

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

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

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        meta = self.agent.init_meta()
        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(
                self.eval_env,
                enabled=(episode == 0 and self.cfg.obs_type in VIDEO_RECORDING_TYPES),
            )
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(
                        time_step.observation, meta, self.global_step, eval_mode=True
                    )
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f"{self.global_frame}.mp4")

        with self.logger.log_and_dump_ctx(self.global_frame, ty="eval") as log:
            log("episode_reward", total_reward / episode)
            log("episode_length", step * self.cfg.action_repeat / episode)
            log("episode", self.global_episode)
            log("step", self.global_step)

    def train(self):
        # predicates
        train_until_step = utils.Until(
            self.cfg.num_train_frames, self.cfg.action_repeat
        )
        seed_until_step = utils.Until(self.cfg.num_seed_frames, self.cfg.action_repeat)
        eval_every_step = utils.Every(
            self.cfg.eval_every_frames, self.cfg.action_repeat
        )

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        meta = self.agent.init_meta()
        self.replay_storage.add(time_step, meta)
        self.train_video_recorder.init(
            time_step.observation, enabled=(self.cfg.obs_type in VIDEO_RECORDING_TYPES),
        )
        metrics = None
        pbar = tqdm(total=self.cfg.num_train_frames)
        while train_until_step(self.global_step):
            pbar.update(1)
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f"{self.global_frame}.mp4")
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(
                        self.global_frame, ty="train"
                    ) as log:
                        log("fps", episode_frame / elapsed_time)
                        log("total_time", total_time)
                        log("episode_reward", episode_reward)
                        log("episode_length", episode_frame)
                        log("episode", self.global_episode)
                        log("buffer_size", len(self.replay_storage))
                        log("step", self.global_step)

                # reset env
                time_step = self.train_env.reset()
                meta = self.agent.init_meta()
                self.replay_storage.add(time_step, meta)
                self.train_video_recorder.init(
                    time_step.observation,
                    enabled=(
                        self.cfg.obs_type in VIDEO_RECORDING_TYPES
                        and self._global_episode % self.cfg.save_train_video_every == 1
                    ),
                )

                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log(
                    "eval_total_time", self.timer.total_time(), self.global_frame
                )
                self.eval()

            meta = self.agent.update_meta(meta, self.global_step, time_step)

            if hasattr(self.agent, "regress_meta"):
                repeat = self.cfg.action_repeat
                every = self.agent.update_task_every_step // repeat
                init_step = self.agent.num_init_steps
                if (
                    self.global_step > (init_step // repeat)
                    and self.global_step % every == 0
                ):
                    meta = self.agent.regress_meta(self.replay_iter, self.global_step)

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(
                    time_step.observation, meta, self.global_step, eval_mode=False
                )

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty="train")

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step, meta)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

        pbar.close()
        self.train_env.close()
        self.eval_env.close()

    def load_snapshot(self):
        print(f"Loading pretrined agent...")
        snapshot_base_dir = Path(self.cfg.snapshot_base_dir)
        domain, _ = self.cfg.task.split("_", 1)
        snapshot_dir = (
            snapshot_base_dir / self.cfg.obs_type / domain / self.cfg.agent.name
        )

        def try_load(seed):
            snapshot = snapshot_dir / str(seed) / f"snapshot_{self.cfg.snapshot_ts}.pt"
            if not snapshot.exists():
                print(f"{snapshot} does not exist!")
                return None
            print(f"Loading {snapshot}...")
            with snapshot.open("rb") as f:
                payload = torch.load(f)
            return payload

        # try to load current seed
        payload = try_load(self.cfg.seed)
        if payload is not None:
            return payload
        # otherwise try random seed
        while True:
            seed = np.random.randint(1, 11)
            payload = try_load(seed)
            if payload is not None:
                return payload
        return None


@hydra.main(config_path=".", config_name="finetune")
def main(cfg):
    from finetune import Workspace as W

    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / "snapshot.pt"
    if snapshot.exists():
        print(f"resuming: {snapshot}")
        workspace.load_snapshot()
    workspace.train()


if __name__ == "__main__":
    main()
