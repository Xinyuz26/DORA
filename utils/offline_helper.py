import sys
sys.path.append('.')
from utils.replay_memory import Memory
import h5py
import os
import json
import torch
import random
import ast
import re
import warnings
import numpy as np
from typing import Dict, List

def set_seed(seed : int = 43, using_cuda : bool = True):
    """
    Seed the different random generators.

    :param seed:
    :param using_cuda:
    """
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(seed)

    if using_cuda:
        # Deterministic operations for CuDNN, it may impact performances
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    

def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys

def data_loader(h5path : str):
    data_dict = {}
    with h5py.File(h5path, 'r') as dataset_file:
        # for k in tqdm(get_keys(dataset_file), desc="load datafile"):
        for k in get_keys(dataset_file):
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]

    return data_dict

def json_to_vec(json_file_path : str):
    assert os.path.exists(json_file_path)
    with open(json_file_path, 'r') as f:
        p = json.load(f)
    f.close()
    vec = []
    for val in p.values():
        vec.append(val)
    vec = np.array(vec).reshape((-1, ))
    
    return vec

def split_dataset_into_trajs(
    dataset: Dict[str, np.ndarray], max_episode_steps: int = 1000
):
    """Split the [D4RL] style dataset into trajectories
    
    :return: the corresponding start index and end index (not included) of every trajectories
    """
    max_steps = dataset["observations"].shape[0]
    terminal_key = 'terminals' if 'terminals' in list(dataset.keys()) else 'dones'
    if "timeouts" in dataset:
        timeout_idx = np.where(dataset["timeouts"] == True)[0] + 1
        terminal_idx = np.where(dataset[terminal_key] == True)[0] + 1
        start_idx = sorted(
            set(
                [0]
                + timeout_idx[timeout_idx < max_steps].tolist()
                + terminal_idx[terminal_idx < max_steps].tolist()
                + [max_steps]
            )
        )
        traj_pairs = list(zip(start_idx[:-1], start_idx[1:]))
    else:
        if max_episode_steps is None:
            raise Exception(
                "You have the specify the max_episode_steps if no timeouts in dataset"
            )
        else:
            traj_pairs = []
            i = 0
            while i < max_steps:
                start_idx = i
                traj_len = 1
                while (traj_len <= max_episode_steps) and (i < max_steps):
                    i += 1
                    traj_len += 1
                    if dataset[terminal_key][i - 1]:
                        break
                traj_pairs.append([start_idx, i])

    return traj_pairs

def json_loader(json_file_path : str):
    assert os.path.exists(json_file_path)
    with open(json_file_path, 'r') as f:
        json_str = f.read()
    p = json.loads(json_str)
    f.close()
    for key in p.keys():
        find_floats = re.findall(r"[-+]?\d*\.\d+e?[-+]?\d*", p[key])
        if len(find_floats) == 1:
            p[key] = float(find_floats[0])
        else:
            p[key] = np.array([float(num) for num in find_floats])
                 
    return p

class ReplayEnv:
    def __init__(
            self,
            dataset : Dict[str, np.ndarray],
            env_params : np.ndarray = None,
            max_traj_length : int = 1000
        ) -> None:
        
        self.observations = dataset['observations']
        self.actions = dataset['actions']
        self.next_observations = dataset['next_observations']
        self.rewards = dataset['rewards']
        self.dones = dataset['dones'] if 'dones' in dataset.keys() else dataset['terminals']
        self.size = self.observations.shape[0]
        self.max_traj_length = max_traj_length
        self.cur_traj_length = 0
        self.cur_step_index = 0
        
        self.env_params = env_params
        self.last_action = np.zeros((self.actions.shape[-1]))
        self.repeat_flag = False
        self.min_return = None
        self.max_return = None
        self.returns = 0
        self.traj_cnt = 0

    def reset(self):
        return self.observations[self.cur_step_index]
    
    def replay_action(self):
        action = self.actions[self.cur_step_index]
        last_action = self.last_action
        self.last_action = action
        return action, last_action
    
    def step(self):
        next_state = self.next_observations[self.cur_step_index]
        reward = self.rewards[self.cur_step_index]
        done = self.dones[self.cur_step_index]
        self.cur_step_index += 1
        self.cur_traj_length += 1
        self.returns += reward
        info = {
            'cur_traj_index' : self.cur_traj_length,
            'cur_dataset_index' : self.cur_step_index,
            'env_params' : self.env_params
        }
        
        if done or self.cur_traj_length == self.max_traj_length or self.cur_step_index == self.size:
            self.cur_traj_length = 0
            if self.min_return is None:
                self.min_return = self.returns
                self.max_return = self.returns
            self.min_return = min(self.returns, self.min_return)
            self.max_return = max(self.returns, self.max_return)
            self.last_action = np.zeros((self.actions.shape[-1]))
            done = True
            self.returns = 0
            self.traj_cnt += 1
        
        if self.cur_step_index == self.size:
            warnings.warn('Replay is over and will repeat again')
            print('=' * 100)
            print(f'TrajNums : {self.traj_cnt} TransitionNums : {self.cur_step_index} MinReturns : {self.min_return} MaxReturns : {self.max_return}')
            print('=' * 100)
            self.cur_step_index = 0
            self.repeat_flag = True
        
        return next_state, reward, done, info

class RandomReplayEnv:
    def __init__(
            self,
            dataset : Dict[str, np.ndarray],
            env_params : np.ndarray = None,
            max_traj_length : int = 1000
            ) -> None:
        
        self.observations = dataset['observations']
        self.actions = dataset['actions']
        self.next_observations = dataset['next_observations']
        self.rewards = dataset['rewards']
        self.dones = dataset['dones'] if 'dones' in dataset.keys() else dataset['terminals']
        self.size = self.observations.shape[0]
        self.max_traj_length = max_traj_length

        self.traj_pairs = split_dataset_into_trajs(dataset)
        random.shuffle(self.traj_pairs)

        self.cur_traj_length = 0
        self.cur_traj_index = 0
        self.cur_step_index = None
        
        self.env_params = env_params
        self.last_action = np.zeros((self.actions.shape[-1]))
        self.repeat_flag = False
        self.min_return = None
        self.max_return = None
        self.returns = 0
        self.traj_cnt = 0

    def reset(self):
        start, end = self.traj_pairs[self.cur_traj_index]
        self.cur_step_index = start + self.cur_traj_length
        return self.observations[self.cur_step_index]
    
    def replay_action(self):
        action = self.actions[self.cur_step_index]
        last_action = self.last_action
        self.last_action = action
        return action, last_action
    
    def step(self):
        
        next_state = self.next_observations[self.cur_step_index]
        reward = self.rewards[self.cur_step_index]
        done = self.dones[self.cur_step_index]

        self.cur_traj_length += 1
        self.cur_step_index += 1

        self.returns += reward

        info = {
            'cur_traj_index' : self.cur_traj_length,
            'env_params' : self.env_params
        }
        
        if done or self.cur_traj_length == self.max_traj_length:
            start, end = self.traj_pairs[self.cur_traj_index]
            if self.cur_traj_length == self.max_traj_length:
                assert start + self.cur_traj_length == end
            
            self.cur_traj_length = 0
            self.cur_step_index = None
            self.cur_traj_index += 1
            
            if self.min_return is None:
                self.min_return = self.returns
                self.max_return = self.returns
            self.min_return = min(self.returns, self.min_return)
            self.max_return = max(self.returns, self.max_return)
            self.last_action = np.zeros((self.actions.shape[-1]))
            done = True
            self.returns = 0
            self.traj_cnt += 1
        
        if self.cur_traj_index == len(self.traj_pairs):
            warnings.warn('Replay is over and will repeat again')
            print('=' * 100)
            print(f'TrajNums : {self.traj_cnt} TransitionNums : {self.cur_step_index} MinReturns : {self.min_return} MaxReturns : {self.max_return}')
            print('=' * 100)
            self.cur_traj_index = 0
            self.repeat_flag = True
        
        return next_state, reward, done, info

class MultiTaskReplayEnv:
    def __init__(
            self,
            datasets : List[Dict[str, np.ndarray]],
            task_params : List,
            max_traj_length : int = 1000,
            use_state_normalize : bool = True,
            ) -> None:
        
        if not use_state_normalize:
            self.state_mu, self.state_std = 0.0, 1.0
        else:
            array = []
            for dataset in datasets:
                array.append(dataset['observations'])
            array = np.concatenate(array, axis = 0)
            self.state_mu, self.state_std = np.mean(array, axis = 0), np.std(array, axis = 0) + 1e-6
            for i in range(len(datasets)):
                datasets[i]['observations'] = (datasets[i]['observations'] - self.state_mu) / self.state_std
                datasets[i]['next_observations'] = (datasets[i]['next_observations'] - self.state_mu) / self.state_std
        
        # origin version
        self.replay_envs = [
            ReplayEnv(dataset, task_param, max_traj_length)
            for dataset, task_param in zip(datasets, task_params)
        ]

        self.size = 0
        for replay_env in self.replay_envs:
            self.size += replay_env.size
            
        self.task_num = len(datasets)
        self.allow_replay_index = list(range(len(datasets)))
        self.state = None
        self.done = None
        self.cnt = 0
        self._reset()

    @property
    def replay_over(self):
        self._check_replay_over()

        return len(self.allow_replay_index) == 0
    
    def _check_replay_over(self):
        for ind in self.allow_replay_index:
            if self.replay_envs[ind].repeat_flag:
                self.allow_replay_index.remove(ind)
    
    def _reset(self):
        # self.cur_env_index = self.allow_replay_index[np.random.choice(len(self.allow_replay_index))]
        self.cur_env_index = self.allow_replay_index[self.cnt % len(self.allow_replay_index)]
        self.cnt += 1
        self.state = self.replay_envs[self.cur_env_index].reset()
        self.done = False
    
    def replay_mem(self) -> Memory:
        action, last_action = self.replay_envs[self.cur_env_index].replay_action()
        next_state, reward, done, info = self.replay_envs[self.cur_env_index].step()
        task_param = info['env_params']
        task_ind = self.cur_env_index
        mask = 0.0 if done else 1.0
        mem = Memory()
        mem.push(
            self.state,
            action,
            [mask],
            next_state,
            [reward],
            None,
            [task_ind + 1],
            task_param,
            last_action,
            [done],
            [1]
        )
        if done:
            if self.replay_over:
                warnings.warn('All transition has been replayed!')
            else:
                self._reset()
        else:
            self.state = next_state
        
        return mem


                