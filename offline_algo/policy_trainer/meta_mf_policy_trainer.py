import time
import os

import numpy as np
import torch
import gym

from typing import Optional, Dict, List
from tqdm import tqdm
from collections import deque
from offline_algo.buffer import ReplayBuffer, OfflineMultiTaskBuffer
from offline_algo.utils.logger import Logger
from offline_algo.policy import BasePolicy
from offline_algo.diy_dataset.rand_wrapper import ParaWrapper
from offline_algo.diy_dataset.ns_wrapper import NS_Wrapper

from models.policy import Policy
from operator import itemgetter


class MetaMFPolicyTrainer:
    def __init__(
        self,
        policy: BasePolicy,
        eval_env: gym.Env,
        buffer: ReplayBuffer,
        logger: Logger,
        obs_dim: int,
        act_dim: int,
        rnn_fix_length: int,
        encoder_path: str,
        epoch: int = 1000,
        step_per_epoch: int = 1000,
        batch_size: int = 256,
        eval_episodes: int = 10,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        task_params : List = None,
    ) -> None:
        self.policy = policy
        self.eval_env = ParaWrapper(eval_env)
        self.ns_env = NS_Wrapper(eval_env, change_interval = 50)
        ns_tasks = self.ns_env.sample_tasks(list(task_params[0].keys()), 10, 1.8)
        self.ns_env.set_ns_params(ns_tasks)
        self.buffer = buffer
        self.logger = logger

        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._batch_size = batch_size
        self._eval_episodes = eval_episodes
        self.lr_scheduler = lr_scheduler

        self.task_params = task_params
        print('eval tasks')
        print(self.task_params)
        print('ns tasks')
        print(ns_tasks)        
        # init encoder
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.rnn_fix_length = rnn_fix_length
        self.encoder_path = encoder_path
        # print(self.obs_dim, self.act_dim )
        self.encoder = Policy(
            obs_dim = self.obs_dim,
            act_dim = self.act_dim,
            up_hidden_size = [128, 64],
            up_activations = ['leaky_relu', 'leaky_relu', 'linear'],
            up_layer_type = ['fc', 'fc', 'fc'],
            ep_hidden_size = [128, 64],
            ep_activation = ['leaky_relu', 'linear', 'tanh'], 
            ep_layer_type = ['fc', 'gru', 'fc'],
            ep_dim = 2,
            use_gt_env_feature = False,
            rnn_fix_length = self.rnn_fix_length,
            logger = None,
            bottle_sigma = 0.01,
            out_ep_std = False
        )

        self._use_ns_eval = False
        
        if not isinstance(buffer, ReplayBuffer):
            self.encoder.to('cuda')
            self.encoder.load(self.encoder_path)
            self.buffer.extend_z_embeddings(self.encoder)
    
    def set_use_ns_eval(self, use_ns_eval : bool = True):
        self._use_ns_eval = use_ns_eval


    def train(self) -> Dict[str, float]:
        start_time = time.time()

        num_timesteps = 0
        last_10_performance = deque(maxlen=10)
        # train loop
        for e in range(1, self._epoch + 1):

            self.policy.train()

            pbar = tqdm(range(self._step_per_epoch), desc=f"Epoch #{e}/{self._epoch}")
            for it in pbar:
                batch = self.buffer.sample(self._batch_size)
                loss = self.policy.learn(batch)
                pbar.set_postfix(**loss)

                for k, v in loss.items():
                    self.logger.logkv_mean(k, v)
                
                num_timesteps += 1

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.logger.set_timestep(num_timesteps)
            self.logger.dumpkvs()
        
            # save checkpoint
            torch.save(self.policy.state_dict(), os.path.join(self.logger.checkpoint_dir, "policy.pth"))

        self.logger.log("total time: {:.2f}s".format(time.time() - start_time))
        torch.save(self.policy.state_dict(), os.path.join(self.logger.model_dir, "policy.pth"))
        self.logger.close()

        return {"last_10_performance": np.mean(last_10_performance)}
    
    def _single_evaluate(self) -> Dict[str, List[float]]:
        self.policy.eval()
        eval_ep_info_buffer = []
        
        for index in range(len(self.task_params)):
            self.eval_env.set_para(self.task_params[index])
            # print("test task index:", index, "  test task para: ", self.task_params[index])
            obs = self.eval_env.reset()
            num_episodes = 0
            episode_reward, episode_length = 0, 0

            while num_episodes < self._eval_episodes:

                action = self.policy.select_action(obs.reshape(1,-1), deterministic=True)
                next_obs, reward, terminal, _ = self.eval_env.step(action.flatten())
                episode_reward += reward
                episode_length += 1

                obs = next_obs
                if terminal:
                    eval_ep_info_buffer.append(
                        {"episode_reward": episode_reward, "episode_length": episode_length}
                    )
                    num_episodes +=1
                    episode_reward, episode_length = 0, 0
                    self.eval_env.set_para(self.task_params[index])
                    obs = self.eval_env.reset()
        
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }

    def _evaluate(self) -> Dict[str, List[float]]:
        self.policy.eval()
        eval_ep_info_buffer = []
        
        for index in range(len(self.task_params)):
            self.eval_env.set_para(self.task_params[index])
            # print("test task index:", index, "  test task para: ", self.task_params[index])
            obs = self.eval_env.reset()
            num_episodes = 0
            episode_reward, episode_length = 0, 0

            all_avg_z = self.buffer.get_all_tasks_z()
            z = torch.from_numpy( all_avg_z[index] ).to('cuda').reshape((1, -1))
            print("task embeddings: " , z)

            while num_episodes < self._eval_episodes:

                action = self.policy.select_action( torch.from_numpy(obs.reshape(1,-1) ).to("cuda:0") , z,  deterministic=True)
                next_obs, reward, terminal, _ = self.eval_env.step(action.flatten())
                episode_reward += reward
                episode_length += 1

                obs = next_obs
                if terminal:
                    eval_ep_info_buffer.append(
                        {"episode_reward": episode_reward, "episode_length": episode_length}
                    )
                    num_episodes +=1
                    episode_reward, episode_length = 0, 0
                    self.eval_env.set_para(self.task_params[index])
                    obs = self.eval_env.reset()
        
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }
    
    def _ns_evaluate(self):
        self.policy.eval()
        eval_ep_info_buffer = []

        for _ in range(20):
            obs = self.ns_env.reset()

            episode_reward, episode_length = 0, 0

            history_traj_length = self.encoder.rnn_fix_length
            history_observations = deque(maxlen = int(history_traj_length * 1.0))
            history_actions = deque(maxlen = int(history_traj_length * 1.0))

            # padding with zero
            for k in range(history_traj_length):
                history_observations.append(np.zeros((self.encoder.obs_dim, )))
                history_actions.append(np.zeros((self.encoder.act_dim, )))
            
            history_observations.append(obs)
            done = False

            while not done:
                # inference z firstly
                with torch.no_grad():
                    history_trajectory = np.concatenate([np.array(history_observations), np.array(history_actions)], axis = -1)
                    history_trajectory = torch.from_numpy(history_trajectory).to(self.encoder.device, dtype = torch.get_default_dtype())
                    z, _ = self.encoder.get_ep(history_trajectory.unsqueeze(0), h = None)
                    z = z[-1, -1, : ].reshape(1,-1)
                    
                
                action = self.policy.select_action(torch.from_numpy(obs.reshape(1, -1)).to(self.encoder.device), z, deterministic = True)[0]
                next_obs, reward, done, info = self.ns_env.step(action.flatten())

                
                episode_reward += reward
                episode_length += 1
                
                history_observations.append(next_obs.copy())
                history_actions.append(action.copy())

                obs = next_obs
            
            eval_ep_info_buffer.append(
                {"episode_reward": episode_reward, "episode_length": episode_length}
            )
        
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }
