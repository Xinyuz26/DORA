from loss.encoder_loss import RMDMLoss
from utils.logger import Logger
from models.policy import Policy
from typing import Dict, NamedTuple, List
from utils.replay_memory import MemoryArray, Memory
from utils.offline_helper import ReplayEnv, data_loader, json_to_vec, MultiTaskReplayEnv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import torch.nn.functional as F
# from collections import deque

from utils.visualize_repre import (
    get_figure,
    visualize_repre,
    visualize_repre_real_param
)

class Embedding(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super(Embedding, self).__init__()
        self.embedings = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.embedings(x)
    
class Trainer:
    def __init__(
            self,
            datasets : List[str],
            task_params : List[str],
            parameter : NamedTuple,
            env_parameter_dict : Dict,
            policy_config : Dict,
            logger : Logger,
            use_embeddings : bool = False,
            merged_token : bool = True,
            ) -> None:
        
        self.parameter = parameter
        self.logger = logger
        self.policy_config = policy_config
        self.policy_config['logger'] = logger
        self.use_embeddings = use_embeddings
        self.embedding_size = 32
        self.merged_token = merged_token

        self.device = (
            torch.device('cuda')
            if torch.cuda.is_available()
            else torch.device('cpu')
        )
        
        if use_embeddings:
            print('Using state embedings and action embeddings!!!')
            obs_dim, action_dim = self.policy_config['obs_dim'], self.policy_config['act_dim']
            self.state_embeddings = nn.Linear(obs_dim, self.embedding_size)
            self.action_embeddings = nn.Linear(action_dim, self.embedding_size)
            self.state_embeddings.to(self.device)
            self.action_embeddings.to(self.device)
            self.policy_config['obs_dim'] = self.embedding_size
            self.policy_config['act_dim'] = self.embedding_size
        
        assert self.merged_token or (not self.merged_token and self.use_embeddings), 'Seprated token must be processed with an additional embedding layer'
        if not self.merged_token:
            # assert self.policy_config['obs_dim'] == self.policy_config['act_dim']
            self.policy_config['obs_dim'] = int(self.embedding_size / 2)
            self.policy_config['act_dim'] = int(self.embedding_size / 2)
        
        self.policy = Policy(**self.policy_config, out_ep_std = False)
        self.target_policy = Policy(**self.policy_config, out_ep_std = False)
            
        self.env_parameter_dict = env_parameter_dict

        self.datasets_path = datasets
        self.task_params_path = task_params
        
        self.policy.to(self.device)
        self.target_policy.to(self.device)

        ## w_1 is the weight of distortion loss(contrastive loss) , w_2 is the weight of debias loss(KL loss) 
        self.w_1 = 1.0
        self.w_2 = 1.0

        with open(os.path.join(self.logger.output_dir, 'loss_weight.txt'), 'w+') as f:
            f.write(f'contrastive : KL = {self.w_1} : {self.w_2}')

        self._init_model()

        # logger some additional config
        self.logger('-' * 30)
        self.logger(f'Use Embedding {self.use_embeddings}')
        self.logger(f'Seperated Token {not self.merged_token}')
        self.logger(f'Constrative / KL  {self.w_1} / {self.w_2}')
        self.abnormal_batch_cnt = 0
    
    def _init_model(self):
        # init replayEnv
        datasets = []
        task_params = []
        for dataset_path, task_param_path in zip(self.datasets_path, self.task_params_path):
            datasets.append(data_loader(dataset_path))
            task_params.append(json_to_vec(task_param_path))
        
        self.state_dim = datasets[-1]['observations'].shape[-1]
        self.action_dim = datasets[-1]['actions'].shape[-1]

        self.replay_envs = MultiTaskReplayEnv(
            datasets,
            task_params, 
            max_traj_length = 1000,
            use_state_normalize = False
        )
        
        self.model_distance = None
        # init buffer
        self.replay_buffer = MemoryArray(
            max_trajectory_num = 10000,
            max_traj_step = 1050,
            fix_length = self.parameter.rnn_fix_length
        )
        # init rmdm loss
        self.rmdm_loss = RMDMLoss(
            model_distance_matrix = self.model_distance,
            tau = self.parameter.rmdm_tau,
            max_env_len = len(self.datasets_path),
            stop_update = self.parameter.stop_update
        )
        self.rmdm_loss.aux_kernel_radius = 200.0
        self.logger(f'Kernel radius : {self.rmdm_loss.aux_kernel_radius}')
        # init opt
        encoder_parameters = [*self.policy.ep.parameters(True)]
        if self.use_embeddings:
            encoder_parameters += [*self.state_embeddings.parameters(True)]
            encoder_parameters += [*self.action_embeddings.parameters(True)]

        self.encoder_opt = torch.optim.Adam(
            encoder_parameters, lr = self.parameter.policy_learning_rate,
        )
        # init weight
        self.log_consis_w_alpha = (
            (
                torch.ones((1)).to(torch.get_default_dtype())
                * np.log(self.parameter.consistency_loss_weight)
            )
            .to(self.device)
            .requires_grad_(True)
        )
        self.log_diverse_w_alpha = (
            (
                torch.ones((1)).to(torch.get_default_dtype())
                * np.log(self.parameter.diversity_loss_weight)
            )
            .to(self.device)
            .requires_grad_(True)
        )
        self.all_repre = None
    
    def train(self):
        self.logger('Init Buffer!')
        
        while self.replay_buffer.size < self.parameter.start_train_num:
            mem = self.replay_envs.replay_mem()
            self.replay_buffer.mem_push_array(mem)

        self.logger('Init Finished')
        for timestep in range(self.replay_envs.size):
        # for timestep in range(total_timesteps):
            if not self.replay_envs.replay_over:
                replay_steps = 1
                for _ in range(replay_steps):
                    if self.replay_envs.replay_over:
                        continue
                    mem = self.replay_envs.replay_mem()
                    self.replay_buffer.mem_push_array(mem)
            batch = self.replay_buffer.sample_fix_length_sub_trajs(
                self.parameter.sac_mini_batch_size, self.parameter.rnn_fix_length
            )
            log = self.update(batch)
            if timestep % 1000 == 0:
                self.update_target_ep()
            for key in log.keys():
                self.logger.tb.add_scalar(key, log[key], timestep)
            # logger
            self.logger.log_tabular(
                'TrajNum', len(self.replay_buffer), tb_prefix = 'Buffer'
            )
            self.logger.log_tabular(
                'TransitionNum', self.replay_buffer.size, tb_prefix = 'Buffer'
            )

            if timestep % 10000 == 0:
                self.plot_repre(iter = timestep / self.parameter.max_iter_num)
            
            if timestep % self.parameter.max_iter_num == 0:
                self.save()
            
            self.logger.dump_tabular(write = False)
    
    def update(self, batch, with_kl_regular : bool = True):
        log = {}
        dtype = torch.get_default_dtype()
        device = self.device
        consis_w = torch.exp(self.log_consis_w_alpha)
        diverse_w = torch.exp(self.log_diverse_w_alpha)
        (
            states,
            next_states,
            actions,
            last_actions,
            rewards,
            masks,
            valid,
            task,
            env_param
        ) = map(
            lambda x : torch.from_numpy(np.array(x)).to(dtype = dtype, device = device).detach(),
            [
                batch.state,
                batch.next_state,
                batch.action,
                batch.last_action,
                batch.reward,
                batch.mask,
                batch.valid,
                batch.task,
                batch.env_param
            ]
        )
        # init hidden state
        policy_hidden = self.policy.make_init_state(
            batch_size = states.shape[0], device = states.device
        )
        data_is_valid = (valid.sum().item() >= 2)
        if data_is_valid:
            ep_h = policy_hidden[ : self.policy.ep.rnn_num]
            if not self.policy.out_ep_std:
                if self.use_embeddings:
                    states = self.state_embeddings(states)
                    last_actions = self.action_embeddings(last_actions)
                
                if not self.merged_token:
                    seperated_input = torch.stack([last_actions, states], dim = 2).view((states.shape[0], -1, states.shape[-1]))
                    _, _ = self.policy.get_ep(seperated_input, ep_h)
                else:
                    _, _ = self.policy.get_ep(torch.cat((states, last_actions), dim = -1), ep_h)
                z_tensor = self.policy.ep_tensor
            else:
                _, mu_1, log_std_1, _ = self.policy.get_rsample_ep(torch.cat((states, last_actions), dim = -1), ep_h)
                z_tensor = self.policy.ep_tensor

            z = z_tensor[..., -1 : , :]
            (
                rmdm_loss_tensor,
                consistency_loss,
                diverse_loss,
                contrastive_loss,
                batch_task_num,
                _,
                _,
                all_repre,
                all_valids,
            ) = self.rmdm_loss.rmdm_loss_timing(
                z,
                task,
                valid,
                consis_w,
                diverse_w,
                True,
                True,
                rbf_radius=self.parameter.rbf_radius,
                kernel_type=self.parameter.kernel_type,
            )
            self.all_repre = [item.detach() for item in all_repre]
            self.all_valids = [item.detach() for item in all_valids]
            self.all_tasks = self.rmdm_loss.lst_tasks

            if with_kl_regular:
                if not self.policy.out_ep_std:
                    if self.merged_token:
                        z_with_hehavior_policy = z_tensor[ : , 2 : , : ]
                        z_no_hehavior_policy = z_tensor[ : , 1 : -1 : ]
                    else:
                        z_with_hehavior_policy = z_tensor[ : , 2 :: 2, : ]
                        z_no_hehavior_policy = z_tensor[ : , 1 :: 2, : ][: , : -1, : ]
                    gauss_prior = True
                    if not gauss_prior:
                        kl_loss = F.mse_loss(z_no_hehavior_policy.detach(), z_with_hehavior_policy)  
                    else:
                        kl_loss = F.mse_loss(z_with_hehavior_policy, torch.zeros_like(z_with_hehavior_policy))
                else:
                    raise NotImplementedError
            else:
                kl_loss = 0.0
            
            if kl_loss < 0.004:
                kl_loss = kl_loss.detach()
            
            total_loss = self.w_1 * contrastive_loss + self.w_2 * kl_loss
            rmdm_loss_tensor = total_loss

            self.encoder_opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters = self.policy.ep.parameters(True), max_norm = 1.0, norm_type = 2)
            self.encoder_opt.step()
            log['Representation/consis_loss'] = consistency_loss.item() if consistency_loss is not None else 0.0
            log['Representation/diverse_loss'] = diverse_loss.item() if diverse_loss is not None else 0.0
            log['Representation/rmdm_loss'] = rmdm_loss_tensor.item() if rmdm_loss_tensor is not None else 0.0
            log['Representation/constrastive_loss'] = contrastive_loss.item() if rmdm_loss_tensor is not None else 0.0
            log['Representation/batch_task_num'] = batch_task_num
            log['Representation/kl_loss'] = kl_loss.item() if not isinstance(kl_loss, float) else kl_loss
            # checker
            param_sum = 0
            for param in self.policy.ep.parameters():
                param_sum += param.data.sum()
                
            log['check/parameters'] = param_sum
        else:
            self.logger('data is not valid!!')
        
        return log
    
    def update_target_ep(self):
        self.target_policy.load_state_dict(self.policy.state_dict())
    
    def plot_repre(self, iter):
        if self.all_repre is not None:
            fig, fig_mean = visualize_repre(
                self.all_repre,
                self.all_valids,
                os.path.join(self.logger.output_dir, "visual.png"),
                self.env_parameter_dict,
                self.all_tasks,
            )
            fig_real_param = visualize_repre_real_param(
                self.all_repre,
                self.all_valids,
                self.all_tasks,
                self.env_parameter_dict,
            )
            if fig:
                self.logger.tb.add_figure("figs/repre", fig, iter)
                self.logger.tb.add_figure("figs/repre_mean", fig_mean, iter)
                self.logger.tb.add_figure(
                    "figs/repre_real", fig_real_param, iter
                )
    
    def save(self):
        self.policy.save(self.logger.model_output_dir)
        torch.save(
            self.encoder_opt.state_dict(),
            os.path.join(self.logger.model_output_dir, "encoder_optim.pt"),
        )
    
    def load(self):
        self.policy.load(self.logger.model_output_dir, map_location=self.device)
        self.encoder_opt.load_state_dict(
            torch.load(
                os.path.join(self.logger.model_output_dir, "encoder_optim.pt"),
                map_location=self.device,
            )
        )

