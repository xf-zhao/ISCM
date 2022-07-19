from manipulatesound.env import manipulation, wrappers
from nicol.env import presentation
import numpy as np
from dm_control.suite.wrappers import action_scale, pixels
from dmc import ExtendedTimeStepWrapper, FrameStackWrapper
from collections import OrderedDict, deque
from dm_env import StepType, specs

TASKS = {
    "pushout": manipulation.PushOutTask,
    "pushoutgolden": manipulation.PushOutGoldenTask,
    "pushoutblue": manipulation.PushOutBlueTask,
    "pushoutmulti": manipulation.PushOutMultiTask,
    "randompushout": manipulation.RandomPushOutTask,
}


def make(
    name,
    obs_type,
    frame_stack,
    seed,
    screen_width,
    screen_height,
    port=None,
    fake_audio=False,
):
    assert obs_type in [
        "states",
        "pixels",
        "spectrogram_simple",
        "spectrogram_fusion",
        "spectrogram_noisy",
        "pixels_tv",
        "spectrogram_simple_tv",
        "spectrogram_fusion_tv",
        "spectrogram_noisy_tv",
    ]
    domain, task_name = name.split("_", 1)
    task = TASKS[task_name]()
    if obs_type == "states":
        env = manipulation.StateEnv(
            task=task, screen_width=screen_width, screen_height=screen_height, port=port
        )
    elif obs_type == "pixels":
        env = manipulation.PixelEnv(
            task=task, screen_width=screen_width, screen_height=screen_height, port=port
        )
        env = wrappers.FrameStackWrapper(env, frame_stack)
    elif obs_type == "pixels_tv":
        env = manipulation.PixelEnv(
            task=task, screen_width=screen_width, screen_height=screen_height, port=port
        )
        env = wrappers.TVWrapper(env)
        env = wrappers.FrameStackWrapper(env, frame_stack)
    elif obs_type in ("spectrogram_fusion", "spectrogram_simple", "spectrogram_noisy"):
        env = manipulation.PixelSpectrogramEnv(
            task=task, port=port, fake_audio=fake_audio
        )
        env = wrappers.SpectrogramFrameStackWrapper(env, frame_stack)
    elif obs_type in (
        "spectrogram_fusion_tv",
        "spectrogram_simple_tv",
        "spectrogram_noisy_tv",
    ):
        env = manipulation.PixelSpectrogramEnv(
            task=task, port=port, fake_audio=fake_audio
        )
        env = wrappers.TVWrapper(env)
        env = wrappers.SpectrogramFrameStackWrapper(env, frame_stack)
    else:
        raise NotImplementedError
    env = wrappers.ActionScaleWrapper(env, minimum=-1.0, maximum=+1.0)
    env = ExtendedTimeStepWrapper(env)
    return env
