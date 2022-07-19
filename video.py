import cv2
from matplotlib import pyplot as plt
from typing import Mapping
import imageio
import numpy as np
import wandb


class VideoRecorder:
    def __init__(self,
                 root_dir,
                 render_size=256,
                 fps=20,
                 camera_id=0,
                 use_wandb=False):
        if root_dir is not None:
            self.save_dir = root_dir / 'eval_video'
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []
        self.camera_id = camera_id
        self.use_wandb = use_wandb

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(env)

    def record(self, env):
        if self.enabled:
            if hasattr(env, 'physics'):
                frame = env.physics.render(height=self.render_size,
                                           width=self.render_size,
                                           camera_id=self.camera_id)
            else:
                frame = env.render()
            self.frames.append(frame)

    def log_to_wandb(self):
        frames = np.transpose(np.array(self.frames), (0, 3, 1, 2))
        fps, skip = 6, 8
        wandb.log({
            'eval/video':
            wandb.Video(frames[::skip, :, ::2, ::2], fps=fps, format="gif")
        })

    def save(self, file_name):
        if self.enabled:
            if self.use_wandb:
                self.log_to_wandb()
            # path = self.save_dir / file_name
            # imageio.mimsave(str(path), self.frames, fps=self.fps)


class VideoRecorderPlus(VideoRecorder):
    def __init__(self, root_dir, render_size=256, fps=20, camera_id=0, use_wandb=False):
        self.spectrograms = []
        super().__init__(root_dir, render_size=render_size, fps=fps, camera_id=camera_id, use_wandb=use_wandb)

    def init(self, env, enabled=True):
        self.spectrograms = []
        return super().init(env, enabled=enabled)

    def record(self, env):
        if self.enabled:
            if hasattr(env, 'physics'):
                frame = env.physics.render(height=self.render_size,
                                           width=self.render_size,
                                           camera_id=self.camera_id)
            else:
                obs = env.render()
                _frame, _spectrogram = obs['pixels'], obs['spectrogram']
            self.frames.append(_frame)
            self.spectrograms.append(_spectrogram)

    def log_to_wandb(self):
        frames = np.transpose(np.array(self.frames), (0, 3, 1, 2))
        spectrograms = np.transpose(np.array(self.spectrograms), (0, 3, 1, 2))
        fps, skip = 6, 8
        wandb.log({
            'eval/video':
            wandb.Video(frames[::skip, :, ::2, ::2], fps=fps, format="gif"),
            'eval/spectrogram':
            wandb.Video(spectrograms[::2, :, ::2, ::2], fps=fps, format="gif"),
        })

    def save(self, file_name):
        if self.enabled:
            if self.use_wandb:
                self.log_to_wandb()
            # imageio.mimsave(str(self.save_dir / file_name), self.frames, fps=self.fps)
            # imageio.mimsave(str(self.save_dir / ('spectrogram_'+file_name)), self.spectrograms, fps=self.fps)


class TrainVideoRecorder:
    def __init__(self,
                 root_dir,
                 render_size=256,
                 fps=20,
                 camera_id=0,
                 use_wandb=False):
        if root_dir is not None:
            self.save_dir = root_dir / 'train_video'
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []
        self.camera_id = camera_id
        self.use_wandb = use_wandb

    def init(self, obs, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(obs)

    def record(self, obs):
        if self.enabled:
            frame = cv2.resize(obs[-3:].transpose(1, 2, 0),
                               dsize=(self.render_size, self.render_size),
                               interpolation=cv2.INTER_CUBIC)
            self.frames.append(frame)

    def log_to_wandb(self):
        frames = np.transpose(np.array(self.frames), (0, 3, 1, 2))
        fps, skip = 6, 8
        wandb.log({
            'train/video':
            wandb.Video(frames[::skip, :, ::2, ::2], fps=fps, format="gif")
        })

    def save(self, file_name):
        if self.enabled:
            if self.use_wandb:
                self.log_to_wandb()
            # path = self.save_dir / file_name
            # imageio.mimsave(str(path), self.frames, fps=self.fps)

class TrainVideoRecorderPlus(TrainVideoRecorder):
    def __init__(self, root_dir, render_size=256, fps=20, camera_id=0, use_wandb=False):
        super().__init__(root_dir, render_size=render_size, fps=fps, camera_id=camera_id, use_wandb=use_wandb)
        self.spectrograms = []

    def init(self, obs, enabled=True):
        self.spectrograms = []
        return super().init(obs, enabled=enabled)

    def record(self, obs):
        if self.enabled:
            _frame, _spectrogram = obs['pixels'][-3:], (obs['spectrogram'][-1] + 1)*255/2
            frame = cv2.resize(_frame.transpose(1, 2, 0),
                               dsize=(self.render_size, self.render_size),
                               interpolation=cv2.INTER_CUBIC)
            self.frames.append(frame)
            spectrogram = cv2.resize(_spectrogram.astype(np.uint8), #.transpose(1, 2, 0),
                               dsize=(self.render_size//2, self.render_size//2),
                               interpolation=cv2.INTER_CUBIC)
            # plt.imshow(spectrogram)
            # plt.show()
            spectrogram = np.expand_dims(spectrogram, axis=2)
            self.spectrograms.append(spectrogram)

    def log_to_wandb(self):
        frames = np.transpose(np.array(self.frames), (0, 3, 1, 2))
        spectrograms = np.transpose(np.array(self.spectrograms), (0, 3, 1, 2))
        fps, skip = 6, 8
        wandb.log({
            'train/video':
            wandb.Video(frames[::skip, :, ::2, ::2], fps=fps, format="gif"),
            'train/spectrogram':
            wandb.Video(spectrograms[::2, :, ::2, ::2], fps=fps, format="gif"),
        })

    def save(self, file_name):
        if self.enabled:
            if self.use_wandb:
                self.log_to_wandb()
            path = self.save_dir / file_name
            # imageio.mimsave(str(path), self.frames, fps=self.fps)
            # imageio.mimsave(str(self.save_dir / ('spectrogram_'+file_name)), self.spectrograms, fps=self.fps)
