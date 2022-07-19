from manipulatesound.env import manipulation, wrappers
from nicol.env import presentation
import numpy as np
from dm_control.suite.wrappers import action_scale, pixels
from dmc import ExtendedTimeStepWrapper, FrameStackWrapper
from collections import OrderedDict, deque
from dm_env import StepType, specs

# @title All `dm_control` imports required for this tutorial
from dm_control.suite.wrappers import action_scale
from dm_env import specs
from absl import logging
from dm_control import mjcf
from dm_control.utils.inverse_kinematics import qpos_from_site_pose
from dm_control.composer import observation
from dm_control.rl import control
import dm_env
import numpy as np
from dm_control.composer.observation import observable
from dm_control import viewer
from tqdm import tqdm
import matplotlib.pyplot as plt
from mujoco_py import load_model_from_xml, MjSim, MjViewer
import collections
import numpy as np
from dm_control.manipulation.shared import observations
import mujoco_py
import os

# The basic mujoco wrapper.
from dm_control import mujoco
from dm_control.rl import control

# Access to enums and MuJoCo library functions.
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings import mjlib

# PyMJCF
from dm_control import mjcf, rl

# Composer high level imports
from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.composer import variation

import os

from dm_control.suite import base

from nicol.env import presentation

TASKS = {
    "show": presentation.Pushing,
    "showrelative": presentation.PushingRelative,
    "hide": presentation.Pushing,
}


def make(
    name, obs_type, frame_stack, seed,
):
    assert obs_type in ["states", "pixels", "pixeljoints"]
    domain, task_name = name.split("_", 1)
    task = TASKS[task_name]()
    if obs_type == "states":
        pass
    elif obs_type == "pixels":
        env = presentation.CustomEnvironment(
            task=task,
            time_limit=20,
            random_state=np.random.RandomState(seed),
            raise_exception_on_physics_error=False,
        )
        env = presentation.IKActionWrapper(env)
        env = wrappers.FrameStackWrapper(env, frame_stack)
    else:
        raise NotImplementedError
    env = action_scale.Wrapper(env, minimum=-1, maximum=1)
    env = ExtendedTimeStepWrapper(env)
    return env
