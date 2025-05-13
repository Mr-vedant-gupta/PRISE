import datetime
import io
import random
import traceback
import utils.misc as utils
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
from torch.utils.data.distributed import DistributedSampler


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


class ReplayBuffer(IterableDataset):
    def __init__(self, replay_dir, max_traj_per_task, max_size, num_workers, nstep,
                 nstep_history, fetch_every,rank=None, world_size=None):
        self._replay_dir = replay_dir if type(replay_dir) == list else [replay_dir]
        self._max_traj_per_task = max_traj_per_task
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._nstep_history = nstep_history
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self.rank = rank
        self.world_size = world_size
        print('Loading Data into CPU Memory')
        self._preload()

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def __len__(self):
        return self._size

    def create_stacked_images(self, imgs):
        T, C, H, W = imgs.shape
        history_length = 3
        stacked_imgs = np.zeros((T, C * history_length, H, W), dtype=imgs.dtype)
        # For each frame in the history
        for h in range(history_length):
            # Calculate the offset (0 for current frame, increasing for previous frames)
            offset = history_length - 1 - h

            # Handle the shifted indices:
            # - For frames that would be before the start, use frame 0
            # - For all other frames, use the appropriate previous frame
            indices = np.maximum(0, np.arange(T) - offset)

            # Place in the corresponding channel position
            stacked_imgs[:, h * C:(h + 1) * C, :, :] = imgs[indices]

        return stacked_imgs

    def _store_episode(self, eps_fn):
        episode = load_episode(eps_fn)
        state = episode['states']
        if state.shape[-1] > 8:
            episode['states'] = np.hstack((state[:, :4], state[:, 18 : 18 + 4]))

        # Do the image stacking and remove the extra element
        episode['imgs'] = self.create_stacked_images(episode['imgs'])[:-1]
        episode['states'] = episode['states'][:-1]


        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            raise Exception("this should not happen")
            # early_eps_fn = self._episode_fns.pop(0)
            # early_eps = self._episodes.pop(early_eps_fn)
            # self._size -= episode_len(early_eps)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len
        return True

    def _preload(self):
        eps_fns = []
        for replay_dir in self._replay_dir:
            eps_fns.extend(utils.choose(sorted(replay_dir.glob('*.npz'), reverse=True), self._max_traj_per_task))
        if len(eps_fns)==0:
            raise ValueError('No episodes found in {}'.format(self._replay_dir))
        for eps_idx, eps_fn in enumerate(eps_fns):
            if self.rank is not None and eps_idx % self.world_size != self.rank:
                continue
            else:
                self._store_episode(eps_fn)
        print(f'Process {self.rank} Loaded {len(self._episode_fns)} Trajectories')
    
    
    def _sample(self):
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1
        # print(f"idx: {idx}")
        # print(f"episode len {episode_len(episode)}")
        # print(f"action shape {episode['actions'].shape}")
        # print(f"obs shape {episode['imgs'].shape}")
        action     = episode['actions'][idx]
        action_seq = [episode['actions'][idx+i] for i in range(self._nstep)]
        
        next_obs, next_state = [], []
        for i in range(self._nstep):
            next_obs.append(episode['imgs'][idx + i][None,:])
            next_state.append(episode['states'][idx + i][None,:])
            
        next_obs = np.vstack(next_obs)
        next_state = np.vstack(next_state)         
        next_obs = (next_obs, next_state)
        
        obs_history, state_history = [], []
        timestep = idx - 1
        ### (o_{t-3}, o_{t-2}, o_{t-1}, o_{t}, 0, 0 ...)
        while timestep >= 0 and len(obs_history)<self._nstep_history:
            obs_history = [episode['imgs'][timestep][None,:]] + obs_history
            state_history     = [episode['states'][timestep][None, :]] + state_history
            timestep -= 1
        
        pad_step = self._nstep_history - len(obs_history)
        obs_history = [episode['imgs'][0][None,:] for i in range(pad_step)] + obs_history
        state_history     = [episode['states'][0][None, :] for i in range(pad_step)] + state_history
        
        obs_history = np.vstack(obs_history)
        state_history = np.vstack(state_history)                   
        obs_history = (obs_history, state_history)
                                 
        if 'token' in episode.keys():
            tok = episode['token'][idx-1]
            return (obs_history, action, tok, action_seq, next_obs)
        else:
            return (obs_history, action, action_seq, next_obs)

    def __iter__(self):
        while True:
            yield self._sample()


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader_dist(replay_dir, max_traj_per_task, max_size, batch_size, num_workers,
                        nstep, nstep_history, rank, world_size):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplayBuffer(replay_dir,
                            max_traj_per_task,
                            max_size_per_worker,
                            num_workers,
                            nstep,
                            nstep_history,
                            fetch_every=1000,
                            rank=rank,
                            world_size=world_size)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=False,
                                         worker_init_fn=_worker_init_fn)
    return loader

