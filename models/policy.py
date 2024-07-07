import os
from typing import List, Union

import numpy as np
import torch

from models.rnn_base import (
    RNNBase,
    rnn_append_hidden_state,
    rnn_get_hidden_length,
    rnn_pop_hidden_state,
)
from utils.logger import Logger


class Policy(torch.nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        up_hidden_size: int,
        up_activations: List[str],
        up_layer_type: List[str],
        ep_hidden_size: int,
        ep_activation: str,
        ep_layer_type: str,
        ep_dim: int,
        use_gt_env_feature: bool,
        rnn_fix_length: int,
        logger: Logger = None,
        bottle_sigma: float = 1e-4,
        out_ep_std : bool = False,
        **kwargs
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.use_gt_env_feature = use_gt_env_feature
        self.bottle_sigma = bottle_sigma
        # aux dim: we add ep to every layer inputs.
        aux_dim = 0
        self.ep_dim = ep_dim
        self.up = RNNBase(
            obs_dim + ep_dim,
            act_dim * 2,
            up_hidden_size,
            up_activations,
            up_layer_type,
            logger,
            aux_dim,
        )
        if not out_ep_std:
            self.ep = RNNBase(
                obs_dim + act_dim,
                ep_dim,
                ep_hidden_size,
                ep_activation,
                ep_layer_type,
                logger,
            )
        else:
            self.ep = RNNBase(
                obs_dim + act_dim,
                2 * ep_dim,
                ep_hidden_size,
                ep_activation,
                ep_layer_type,
                logger
            )
        
        self.out_ep_std = out_ep_std
        self.ep_rnn_count = self.ep.rnn_num
        self.up_rnn_count = self.up.rnn_num
        self.module_list = torch.nn.ModuleList(
            self.up.total_module_list
            + self.ep.total_module_list
        )
        self.soft_plus = torch.nn.Softplus()
        self.min_log_std = -7.0
        self.max_log_std = 2.0
        self.sample_hidden_state = None
        self.rnn_fix_length = rnn_fix_length
        self.ep_tensor = None
        self.allow_sample = True
        self.device = torch.device("cpu")

    def set_deterministic_ep(self, deterministic: bool):
        self.allow_sample = not deterministic

    def to(self, device: Union[str, torch.device]):
        if not device == self.device:
            self.device = device
            if self.sample_hidden_state is not None:
                for i in range(len(self.sample_hidden_state)):
                    if self.sample_hidden_state[i] is not None:
                        self.sample_hidden_state[i] = self.sample_hidden_state[i].to(
                            self.device
                        )
            super().to(device)

    def get_ep(
        self, x: torch.Tensor, h: torch.Tensor, require_full_output: bool = False
    ):
        if require_full_output:
            ep, h, full_hidden = self.ep.meta_forward(x, h, require_full_output)
            self.ep_tensor = ep
            return ep, h, full_hidden
        ep, h = self.ep.meta_forward(x, h, require_full_output)
        self.ep_tensor = ep
        return ep, h
    
    def get_rsample_ep(
        self, x : torch.Tensor, h : torch.Tensor
    ):
        assert self.out_ep_std
        total, h = self.ep.meta_forward(x, h)
        mu, log_std = torch.chunk(total, 2, dim = -1)
        log_std = torch.clamp(log_std, min = -20.0, max = 2.0)
        std = log_std.exp()
        noise = torch.randn_like(mu).detach() * std
        ep = mu + noise
        self.ep_tensor = ep

        return ep, mu, log_std, h

    def ep_h(self, h: torch.Tensor):
        return h[: self.ep_rnn_count]

    def up_h(self, h: torch.Tensor):
        return h[self.ep_rnn_count :]

    def make_init_state(self, batch_size: int, device: Union[str, torch.device]):
        ep_h = self.ep.make_init_state(batch_size, device)
        up_h = self.up.make_init_state(batch_size, device)
        h = ep_h + up_h
        return h

    def make_init_action(self, device: Union[str, torch.device] = torch.device("cpu")):
        return torch.zeros((1, self.act_dim), device=device)

    def meta_forward(
        self,
        x: torch.Tensor,
        lst_a: torch.Tensor,
        h: torch.Tensor,
        require_full_output: bool = False,
        ep_out : torch.Tensor = None
    ):
        ep_h = h[: self.ep_rnn_count]
        up_h = h[self.ep_rnn_count :]
        if not require_full_output:
            if not self.use_gt_env_feature and ep_out is None:
                ep, ep_h_out = self.get_ep(torch.cat((x, lst_a), -1), ep_h)
                if self.allow_sample:
                    ep = ep + torch.randn_like(ep) * self.bottle_sigma
                ep = ep.detach()
                up, up_h_out = self.up.meta_forward(torch.cat((x, ep), -1), up_h)
            elif not self.use_gt_env_feature and ep_out is not None:
                assert not ep_out.requires_grad
                ep_h_out = []
                ep = ep_out
                up, up_h_out = self.up.meta_forward(torch.cat((x, ep), -1), up_h)
            else:
                up, up_h_out = self.up.meta_forward(x, up_h)
                ep_h_out = []
        else:
            if not self.use_gt_env_feature:
                ep, ep_h_out, ep_full_hidden = self.get_ep(
                    torch.cat((x, lst_a), -1), ep_h, require_full_output
                )
                if self.allow_sample:
                    ep = ep + torch.randn_like(ep) * self.bottle_sigma
                ep = ep.detach()
                up, up_h_out, up_full_hidden = self.up.meta_forward(
                    torch.cat((x, ep), -1), up_h, require_full_output
                )
            elif not self.use_gt_env_feature and ep_out is not None:
                assert not ep_out.requires_grad
                ep_h_out = []
                ep_full_hidden = []
                ep = ep_out
                up, up_h_out = self.up.meta_forward(torch.cat((x, ep), -1), up_h)
            else:
                up, up_h_out, up_full_hidden = self.up.meta_forward(
                    x, up_h, require_full_output
                )
                ep_h_out = []
                ep_full_hidden = []
            h_out = ep_h_out + up_h_out
            return up, h_out, ep_full_hidden + up_full_hidden
        h_out = ep_h_out + up_h_out
        return up, h_out

    def forward(
        self,
        x: torch.Tensor,
        lst_a: torch.Tensor,
        h: torch.Tensor,
        require_log_std: bool = False,
        ep_out : torch.Tensor = None,
    ):
        policy_out, h_out = self.meta_forward(x, lst_a, h, ep_out = ep_out)
        mu, log_std = policy_out.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        std = log_std.exp()
        if require_log_std:
            return mu, std, log_std, h_out
        return mu, std, h_out

    def rsample(self, x: torch.Tensor, lst_a: torch.Tensor, h: torch.Tensor, ep_out : torch.Tensor = None):
        mu, std, log_std, h_out = self.forward(x, lst_a, h, require_log_std=True, ep_out = ep_out)
        noise = torch.randn_like(mu).detach() * std
        sample = noise + mu
        log_prob = (
            -0.5 * (noise / std).pow(2) - (log_std + 0.5 * np.log(2 * np.pi))
        ).sum(-1, keepdim=True)

        log_prob = log_prob - (
            2 * (-sample - self.soft_plus(-2 * sample) + np.log(2))
        ).sum(-1, keepdim=True)
        return torch.tanh(mu), std, torch.tanh(sample), log_prob, h_out

    def save(self, path: str):
        self.up.save(os.path.join(path, "universe_policy.pt"))
        self.ep.save(os.path.join(path, "environment_probe.pt"))

    def load(self, path: str, **kwargs):
        self.up.load(os.path.join(path, "universe_policy.pt"), **kwargs)
        self.ep.load(os.path.join(path, "environment_probe.pt"), **kwargs)

    def inference_init_hidden(
        self, batch_size: int, device: Union[str, torch.device] = torch.device("cpu")
    ):
        if self.rnn_fix_length is None or self.rnn_fix_length == 0:
            self.sample_hidden_state = self.make_init_state(batch_size, device)
        else:
            self.sample_hidden_state = [None] * len(
                self.make_init_state(batch_size, device)
            )

    def inference_check_hidden(self, batch_size: int):
        if self.sample_hidden_state is None:
            return False
        if len(self.sample_hidden_state) == 0:
            return True
        if self.rnn_fix_length is not None and self.rnn_fix_length > 0:
            return True
        if isinstance(self.sample_hidden_state[0], tuple):
            return self.sample_hidden_state[0][0].shape[0] == batch_size
        else:
            return self.sample_hidden_state[0].shape[0] == batch_size

    def inference_rnn_fix_one_action(
        self, state: torch.Tensor, lst_action: torch.Tensor
    ):
        if self.use_gt_env_feature:
            mu, std, act, logp, self.sample_hidden_state = self.rsample(
                state, lst_action, self.sample_hidden_state
            )
            return mu, std, act, logp, self.sample_hidden_state

        while rnn_get_hidden_length(self.sample_hidden_state) >= self.rnn_fix_length:
            self.sample_hidden_state = rnn_pop_hidden_state(self.sample_hidden_state)
        self.sample_hidden_state = rnn_append_hidden_state(
            self.sample_hidden_state, self.make_init_state(1, state.device)
        )
        if len(state.shape) == 2:
            state = state.unsqueeze(0)
            lst_action = lst_action.unsqueeze(0)
        length = rnn_get_hidden_length(self.sample_hidden_state)
        state = torch.cat([state] * length, dim=0)
        lst_action = torch.cat([lst_action] * length, dim=0)
        mu, std, act, logp, self.sample_hidden_state = self.rsample(
            state, lst_action, self.sample_hidden_state
        )

        return mu, std, act, logp, self.sample_hidden_state

    def inference_one_step(self, state, deterministic=True):
        self.set_deterministic_ep(deterministic)
        with torch.no_grad():
            lst_action = state[..., : self.act_dim]
            state = state[..., self.act_dim :]
            if (
                self.rnn_fix_length is None
                or self.rnn_fix_length == 0
                or len(self.sample_hidden_state) == 0
            ):
                mu, std, act, logp, self.sample_hidden_state = self.rsample(
                    state, lst_action, self.sample_hidden_state
                )
            else:
                while (
                    rnn_get_hidden_length(self.sample_hidden_state)
                    < self.rnn_fix_length - 1
                    and not self.use_gt_env_feature
                ):
                    (
                        _,
                        _,
                        _,
                        _,
                        self.sample_hidden_state,
                    ) = self.inference_rnn_fix_one_action(
                        torch.zeros_like(state), torch.zeros_like(lst_action)
                    )
                (
                    mu,
                    std,
                    act,
                    logp,
                    self.sample_hidden_state,
                ) = self.inference_rnn_fix_one_action(state, lst_action)
                mu, std, act, logp = map(
                    lambda x: x[:1].reshape((1, -1)), [mu, std, act, logp]
                )
        if deterministic:
            return mu
        return act

    def inference_reset_one_hidden(self, idx: int):
        if self.rnn_fix_length is not None and self.rnn_fix_length > 0:
            raise NotImplementedError(
                "if rnn fix length is set, parallel sampling is not allowed!!!"
            )
        for i in range(len(self.sample_hidden_state)):
            if isinstance(self.sample_hidden_state[i], tuple):
                self.sample_hidden_state[i][0][0, idx] = 0
                self.sample_hidden_state[i][1][0, idx] = 0
            else:
                self.sample_hidden_state[i][0, idx] = 0


def policy_hidden_state_slice(hidden_state: torch.Tensor, start: int, end: int):
    res_hidden = []
    len_hidden = len(hidden_state)
    for i in range(len_hidden):
        h = hidden_state[i][:, start:end]
        hid = h
        res_hidden.append(hid)
    return res_hidden


def policy_hidden_detach(hidden_state: torch.Tensor):
    res_hidden = []
    len_hidden = len(hidden_state)
    for i in range(len_hidden):
        res_hidden.append(hidden_state[i].detach())
    return res_hidden
