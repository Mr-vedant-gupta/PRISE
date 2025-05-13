# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import deque
import numpy as np
import gym
from gym.wrappers import TimeLimit
#from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from metaworld import MT1
import dm_env
from dm_env import specs
from typing import Any, NamedTuple
from dm_control.suite.wrappers import action_scale
import mujoco_py
import gc
import cv2
import os
from PIL import Image

# class ExtendedTimeStep(NamedTuple):
#     done: Any
#     reward: Any
#     discount: Any
#     observation: Any
#     state: Any
#     action: Any
#     success: Any
#
#     def last(self):
#         return self.done
#
#     def __getitem__(self, attr):
#         if isinstance(attr, str):
#             return getattr(self, attr)
#         else:
#             return tuple.__getitem__(self, attr)

# class ExtendedTimeStepWrapper(dm_env.Environment):
#     def __init__(self, env):
#         self._env = env
#
#     def reset(self):
#         obs = self._env.reset()
#         return self._augment_time_step(np.array(obs), self.prop_state())
#
#     def step(self, action):
#         obs, reward, done, extra = self._env.step(action)
#         discount = 1.0
#         success=extra['success']
#         return self._augment_time_step(np.array(obs),
#                                        self.prop_state(),
#                                        action,
#                                        reward,
#                                        success,
#                                        discount,
#                                        done)
#     def prop_state(self):
#         state = self._env.state
#         #return state
#         return np.concatenate((state[:4], state[18 : 18 + 4]))
#
#
#     def _augment_time_step(self, obs, state, action=None, reward=None, success=False, discount=1.0, done=False):
#         if action is None:
#             action_spec = self.action_spec()
#             action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
#             reward = 0.0
#             success = 0.0
#             discount = 1.0
#             done = False
#         return ExtendedTimeStep(observation=obs,
#                                 state=state,
#                                 action=action,
#                                 reward=reward,
#                                 success=success,
#                                 discount=discount,
#                                 done = done)
#
#     def state_spec(self):
#         return specs.BoundedArray((8,), np.float32, name='state', minimum=0, maximum=255)
#
#     def observation_spec(self):
#         return specs.BoundedArray(self._env.observation_space.shape, np.uint8, name='observation', minimum=0, maximum=255)
#
#     def action_spec(self):
#         return specs.BoundedArray(self._env.action_space.shape, np.float32, name='action', minimum=-1, maximum=1.0)
#
#     def __getattr__(self, name):
#         return getattr(self._env, name)

class MetaWorldWrapper(gym.Wrapper):
    def __init__(self, env, frame_stack):
        super().__init__(env)
        self.env = env
        self._num_frames = frame_stack
        self._frames = deque([], maxlen=self._num_frames)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._num_frames * 3, 84, 84),
            dtype=np.uint8,
        )
        self.action_space = self.env.action_space
        self._res = (84, 84)
        self.img_size = 84


        
    @property
    def state(self):
        state = self._state_obs.astype(np.float32)
        return state

    def _stacked_obs(self):
        assert len(self._frames) == self._num_frames
        return np.concatenate(list(self._frames), axis=0)
    
    def _downsample_image(self, img):
        assert len(img.shape) == 3
        img = np.array(Image.fromarray(img).transpose(method=Image.FLIP_TOP_BOTTOM))
        new_img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_LINEAR)
        return np.transpose(new_img, (2, 0, 1))

    def prop_state(self):
        state = self.state
        return np.concatenate((state[:4], state[18 : 18 + 4]))

    def get_data(self):
        return {"imgs": self._stacked_obs(),
                "robot_states": self.prop_state()}

    
    def reset(self):
        state, info = self.env.reset()
        self._state_obs = state
        img = self.env.render()
        pixel_obs = self._downsample_image(img)
        for _ in range(self._num_frames):
            self._frames.append(pixel_obs)
        return self.get_data(), 0, False, {"success": False}

    def step(self, action):
        reward = 0
        for _ in range(1):
            state, r, terminal, truncated, info = self.env.step(action)
            reward += r
        self._state_obs = state
        img = self.env.render()
        img = self._downsample_image(img)
        self._frames.append(img)
        return self.get_data(), 0, terminal or truncated, info

    # def render(self, mode="rgb_array", width=None, height=None, camera_id=None):
    #     return self.env.render(offscreen=False, resolution=(width, height), camera_name=self.camera_name).copy()

    def terminate(self):
        self.env.close()
        gc.collect()
        pass

    def observation_spec(self):
        return self.observation_space

    def action_spec(self):
        return self.action_space

    def __getattr__(self, name):
        return getattr(self._env, name)

def mw_gym_make(task_name, task_id=0, seed=None):
    if seed is not None:
        mt1 = MT1(task_name, seed=seed)
    else:
        mt1 = MT1(task_name) # Construct the benchmark, sampling tasks
    env = mt1.train_classes[task_name](render_mode='rgb_array')

    if task_id is not None:
        env.set_task(mt1.train_tasks[task_id])
    return env


def make(name, task_id, frame_stack):
    env = mw_gym_make(name, task_id, seed = 0)
    env = MetaWorldWrapper(env, frame_stack)
    #env = TimeLimit(env, max_episode_steps=600)
    # env = ExtendedTimeStepWrapper(env)
    # env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    env.task_name = name
    return env

