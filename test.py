import td
import numpy as np

name = "manipulation_pushoutmulti"
name = "manipulation_pushoutblue"
env = td.make(name, "pixels", 3, seed=0, screen_width=512, screen_height=512, port=9999)
env.reset()
action = np.random.uniform(env.action_spec().shape)
action = [
    np.random.randn()*0.5,
    np.random.randn()*0.5,
    np.random.randn()*0.5,
    np.random.randn(),]


for i in range(100):
    env.step(action)
