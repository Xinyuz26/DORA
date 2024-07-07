import numpy as np
import torch

from typing import Optional, Union, Tuple, Dict, List
from offline_algo.utils.helper import split_dataset_into_trajs
from models.policy import Policy
from collections import deque
from tqdm import tqdm

class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple,
        obs_dtype: np.dtype,
        action_dim: int,
        action_dtype: np.dtype,
        device: str = "cpu"
    ) -> None:
        self._max_size = buffer_size
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.action_dim = action_dim
        self.action_dtype = action_dtype

        self._ptr = 0
        self._size = 0

        self.observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.next_observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.actions = np.zeros((self._max_size, self.action_dim), dtype=action_dtype)
        self.rewards = np.zeros((self._max_size, 1), dtype=np.float32)
        self.terminals = np.zeros((self._max_size, 1), dtype=np.float32)

        self.device = torch.device(device)
        self.z_buffer = None
        self.z_mean = None

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        terminal: np.ndarray
    ) -> None:
        # Copy to avoid modification by reference
        self.observations[self._ptr] = np.array(obs).copy()
        self.next_observations[self._ptr] = np.array(next_obs).copy()
        self.actions[self._ptr] = np.array(action).copy()
        self.rewards[self._ptr] = np.array(reward).copy()
        self.terminals[self._ptr] = np.array(terminal).copy()

        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)
    
    def add_batch(
        self,
        obss: np.ndarray,
        next_obss: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray
    ) -> None:
        batch_size = len(obss)
        indexes = np.arange(self._ptr, self._ptr + batch_size) % self._max_size

        self.observations[indexes] = np.array(obss).copy()
        self.next_observations[indexes] = np.array(next_obss).copy()
        self.actions[indexes] = np.array(actions).copy()
        self.rewards[indexes] = np.array(rewards).copy()
        self.terminals[indexes] = np.array(terminals).copy()

        self._ptr = (self._ptr + batch_size) % self._max_size
        self._size = min(self._size + batch_size, self._max_size)
    
    def load_dataset(self, dataset: Dict[str, np.ndarray]) -> None:
        observations = np.array(dataset["observations"], dtype=self.obs_dtype)
        next_observations = np.array(dataset["next_observations"], dtype=self.obs_dtype)
        actions = np.array(dataset["actions"], dtype=self.action_dtype)
        rewards = np.array(dataset["rewards"], dtype=np.float32).reshape(-1, 1)
        key = 'terminals' if 'terminals' in dataset.keys() else 'dones'
        terminals = np.array(dataset[key], dtype=np.float32).reshape(-1, 1)

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals

        self._ptr = len(observations)
        self._size = len(observations)
     
    def normalize_obs(self, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.observations.mean(0, keepdims=True)
        std = self.observations.std(0, keepdims=True) + eps
        self.observations = (self.observations - mean) / std
        self.next_observations = (self.next_observations - mean) / std
        obs_mean, obs_std = mean, std
        return obs_mean, obs_std

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:

        batch_indexes = np.random.randint(0, self._size, size=batch_size)
        
        if self.z_buffer is None:
            return {
                "observations": torch.tensor(self.observations[batch_indexes]).to(self.device),
                "actions": torch.tensor(self.actions[batch_indexes]).to(self.device),
                "next_observations": torch.tensor(self.next_observations[batch_indexes]).to(self.device),
                "terminals": torch.tensor(self.terminals[batch_indexes]).to(self.device),
                "rewards": torch.tensor(self.rewards[batch_indexes]).to(self.device)
            }
        else:
            return {
                "observations": torch.tensor(self.observations[batch_indexes]).to(self.device),
                "actions": torch.tensor(self.actions[batch_indexes]).to(self.device),
                "next_observations": torch.tensor(self.next_observations[batch_indexes]).to(self.device),
                "terminals": torch.tensor(self.terminals[batch_indexes]).to(self.device),
                "rewards": torch.tensor(self.rewards[batch_indexes]).to(self.device),
                "z_embeddings" : torch.tensor(self.z_buffer[batch_indexes]).to(self.device),
                "z_next_embeddings" : torch.tensor(self.z_next_buffer[batch_indexes]).to(self.device),
                "z_mean" : torch.tensor(self.z_mean).to(self.device).repeat(batch_size, 1)
            }
    
    def sample_all(self) -> Dict[str, np.ndarray]:
        return {
            "observations": self.observations[:self._size].copy(),
            "actions": self.actions[:self._size].copy(),
            "next_observations": self.next_observations[:self._size].copy(),
            "terminals": self.terminals[:self._size].copy(),
            "rewards": self.rewards[:self._size].copy()
        }
    
    ### extend ###
    def extend_z_embeddings(self, encoder_model : Policy):
        z_dim = encoder_model.ep_dim
        history_traj_length = encoder_model.rnn_fix_length
        self.z_buffer = np.zeros((self._size, z_dim))
        self.z_next_buffer = np.zeros((self._size, z_dim))
        history_observations = deque(maxlen = history_traj_length)
        history_actions = deque(maxlen = history_traj_length)
        history_next_observations = deque(maxlen = history_traj_length)
        history_next_actions = deque(maxlen = history_traj_length)
        # padding with zero
        for k in range(history_traj_length):
            history_observations.append(np.zeros((self.obs_shape[-1], )))
            history_actions.append(np.zeros((self.action_dim, )))
            history_next_observations.append(np.zeros((self.obs_shape[-1], )))
            history_next_actions.append(np.zeros((self.action_dim, )))
        
        # start inference
        last_action = np.zeros((self.action_dim, ))
        for k in range(self._size):
            obs, act, next_obs, done = self.observations[k], self.actions[k], self.next_observations[k], self.terminals[k]
            history_observations.append(obs.copy())
            history_actions.append(last_action.copy())
            history_next_observations.append(next_obs.copy())
            history_next_actions.append(act.copy())

            history_trajectory = np.concatenate([np.array(history_observations), np.array(history_actions)], axis = -1)
            history_trajectory = torch.from_numpy(history_trajectory).to(encoder_model.device, dtype = torch.get_default_dtype())
            history_next_trajectory = np.concatenate([np.array(history_next_observations), np.array(history_next_actions)], axis = -1)
            history_next_trajectory = torch.from_numpy(history_next_trajectory).to(encoder_model.device, dtype = torch.get_default_dtype())

            with torch.no_grad():
                z, _ = encoder_model.get_ep(history_trajectory.unsqueeze(0), h = None)
                z = z[-1, -1, : ]
                z_next, _ = encoder_model.get_ep(history_next_trajectory.unsqueeze(0), h = None)
                z_next = z_next[-1, -1, : ]
            ### debug to check the shape of z
            self.z_buffer[k] = z.detach().cpu().numpy()
            self.z_next_buffer[k] = z_next.detach().cpu().numpy()

            last_action = act
            ### debug to check the type of done
            if done:
                # padding with zero
                for k in range(history_traj_length):
                    history_observations.append(np.zeros((self.obs_shape[-1], )))
                    history_actions.append(np.zeros((self.action_dim, )))
                    history_next_observations.append(np.zeros((self.obs_shape[-1], )))
                    history_next_actions.append(np.zeros((self.action_dim, )))
        
        self.z_mean = np.mean(self.z_buffer, axis = 0, keepdims = True)


class OfflineMultiTaskBuffer:
    def __init__(self,
        datasets : List[Dict],
        device : str = 'cpu',
        task_params : List = None             
    ) -> None:
        buffer_size = datasets[0]['observations'].shape[-1]
        obs_shape = datasets[0]['observations'].shape
        obs_dtype = datasets[0]['observations'].dtype
        action_dim = datasets[0]['actions'].shape[-1]
        action_dtype = datasets[0]['actions'].dtype
        self.buffers = [
            ReplayBuffer(
                buffer_size, obs_shape, obs_dtype, action_dim, action_dtype, device
            )
            for _ in range(len(datasets))
        ]
        for i, dataset in enumerate(datasets):
            self.buffers[i].load_dataset(dataset)
        
        # task id
        self.task_index = []
        if task_params is not None:
            assert len(task_params) == len(datasets)
            for task_param in task_params:
                self.task_index.append(torch.tensor(task_param).to(device = device))
        else:
            z_matrix = torch.tensor(np.eye(len(datasets))).to(device = device)
            for i in range(len(datasets)):
                self.task_index.append(z_matrix[i])
        
        self.task_nums = len(datasets)

    def extend_z_embeddings(self, encoder_model : Policy):
        
        # inference z represtation for each transition
        print('Inference z for each transiton in buffers...')
        
        with tqdm(total = self.task_nums) as t:
            for i in range(self.task_nums):
                self.buffers[i].extend_z_embeddings(encoder_model)
                t.update(1)    
        
        print('End inference')
    
    def sample(self, batch_size: int, sample_inference_z : bool = True) -> Dict[str, torch.Tensor]:
        sample_task_indices = np.random.choice(np.arange(len(self.buffers)), size = batch_size)
        task_sampled_size = [np.sum(sample_task_indices == i) for i in range(len(self.buffers))]
        results = dict(
            observations = [],
            actions = [],
            next_observations = [],
            terminals = [],
            rewards = [],
            z_embeddings = [],
            z_next_embeddings = [],
            z_mean = [],
        )
        for i in range(len(self.buffers)):
            if task_sampled_size[i] == 0:
                continue
            
            batch = self.buffers[i].sample(task_sampled_size[i])
            for key in batch.keys():
                results[key].append(batch[key])
            
            if not sample_inference_z:
                results['z_embeddings'] += [self.task_index[i] for _ in range(task_sampled_size[i])]
                results['z_mean'] += [self.task_index[i] for _ in range(task_sampled_size[i])]
        
        for key in results.keys():
            results[key] = torch.vstack(results[key])
        
        return results
    
    def get_all_tasks_z(self):
        
        return {index : self.buffers[index].z_mean for index in range(self.task_nums)}