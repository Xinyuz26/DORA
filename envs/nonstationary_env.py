import copy

import gym
import numpy as np
from gym import Wrapper
import random

# https://github.com/dennisl88/rand_param_envs


class NonstationaryEnv(Wrapper):
    RAND_PARAMS = [
        "body_mass",
        "gravity",
        "dof_damping_1_dim",
    ]

    def __init__(self, env, rand_params=["gravity"], log_scale_limit=3.0):
        super().__init__(env)
        self.is_diy_env = hasattr(env, "diy_env")
        if len(rand_params) == 1 and rand_params[0] == "None":
            rand_params = []
        if self.is_diy_env:
            rand_params = []
        self.normalize_context = True
        self.log_scale_limit = log_scale_limit
        self.rand_params = rand_params
        self.save_parameters()
        self.min_param, self.max_param = self.get_minmax_parameter(log_scale_limit)
        self.cur_parameter_vector = self.env_parameter_vector_
        self.cur_step_ind = 0

        # for non-stationary changing
        self.setted_env_params = None
        self.setted_env_changing_period = None
        self.setted_env_changing_interval = None
        self.min_action = env.action_space.low
        self.max_action = env.action_space.high
        self.range_action = self.max_action - self.min_action
        self._debug_state = None

    def get_minmax_parameter(self, log_scale_limit: float):
        min_param = {}
        max_param = {}
        bound = lambda x, y: np.array(1.5) ** (
            np.ones(shape=x) * ((-1 if y == "low" else 1) * log_scale_limit)
        )

        if "body_mass" in self.rand_params:
            min_multiplyers = bound((1,), "low")
            max_multiplyers = bound((1,), "high")
            min_param["body_mass"] = np.array([min_multiplyers])
            max_param["body_mass"] = np.array([max_multiplyers])

        if "dof_damping_1_dim" in self.rand_params:
            min_multiplyers = bound((1,), "low")
            max_multiplyers = bound((1,), "high")
            min_param["dof_damping_1_dim"] = np.array([min_multiplyers])
            max_param["dof_damping_1_dim"] = np.array([max_multiplyers])

        if "gravity" in self.rand_params:
            min_multiplyers = bound(self.model.opt.gravity.shape, "low")
            max_multiplyers = bound(self.model.opt.gravity.shape, "high")
            min_param["gravity"] = self.init_params["gravity"] * min_multiplyers
            max_param["gravity"] = self.init_params["gravity"] * max_multiplyers

            if "gravity_angle" in self.rand_params:
                min_param["gravity"][:2] = min_param["gravity"][2]
                max_param["gravity"][:2] = max_param["gravity"][2]

        for key in min_param:
            min_it = min_param[key]
            max_it = max_param[key]
            min_real = np.min([min_it, max_it], 0)
            max_real = np.max([max_it, min_it], 0)
            min_param[key] = min_real
            max_param[key] = max_real
        return min_param, max_param

    def denormalization(self, action):
        return (action + 1) / 2 * self.range_action + self.min_action

    def normalization(self, action):
        return (action - self.min_action) / self.range_action * 2 - 1

    def step(self, action):
        self.cur_step_ind += 1
        if (
            self.setted_env_params is not None
            and self.cur_step_ind % self.setted_env_changing_interval == 0
        ):
            assert isinstance(self.setted_env_params, list)
            env_to_be = {}
            weight_origin = self.cur_step_ind / self.setted_env_changing_period
            weight_in_duration = weight_origin - (weight_origin // 2 * 2)
            ind = int(weight_origin)
            if isinstance(self.setted_env_params[0], dict):
                for key in self.setted_env_params[0]:
                    env_to_be[key] = self.setted_env_params[ind][key]
            elif isinstance(self.setted_env_params[0], int):
                env_to_be = self.setted_env_params[ind]
            elif isinstance(self.setted_env_params[0], list):
                env_to_be = copy.deepcopy(self.setted_env_params[ind])
            else:
                raise NotImplementedError(
                    f"type of {type(self.setted_env_params[ind])} is not implemented."
                )
            self.set_task(env_to_be)
        try:
            res = super(NonstationaryEnv, self).step(action)
            self._debug_state = res[0]
            return res
        except Exception as e:
            print(e)
            return self._debug_state, 0, True, {}

    def set_nonstationary_para(
        self, setting_env_params, changine_period: int, changing_interval: int
    ):
        self.setted_env_changing_period = changine_period
        self.setted_env_params = setting_env_params
        self.setted_env_changing_interval = changing_interval

    def reset_nonstationary(self):
        self.set_nonstationary_para(None, None, None)

    def reset(self, **kwargs):
        self.cur_step_ind = 0
        return super(NonstationaryEnv, self).reset(**kwargs)

    def sample_tasks(self, n_tasks: int, dig_range=None, linspace=False):
        """
        Generates randomized parameter sets for the mujoco env
        Args:
            n_tasks (int) : number of different meta-tasks needed
        Returns:
            tasks (list) : an (n_tasks) length list of tasks
        """

        if self.is_diy_env:
            return self.env.sample_tasks(n_tasks)

        current_task_count_ = 0
        param_sets = []
        if dig_range is None:
            if linspace:

                def uniform_function(low_, up_, size):
                    res = [0] * np.prod(size)
                    interval = (up_ - low_) / (n_tasks - 1)
                    for i in range(len(res)):
                        res[i] = interval * current_task_count_ + low_
                    res = np.array(res).reshape(size)
                    return res

                uniform = uniform_function
            else:
                uniform = lambda low_, up_, size: np.random.uniform(
                    low_, up_, size=size
                )
        else:
            dig_range = np.abs(dig_range)

            def uniform_function(low_, up_, size):
                res = [0] * np.prod(size)
                for i in range(len(res)):
                    if linspace:
                        if current_task_count_ >= n_tasks // 2:
                            interval = (up_ - dig_range) / (n_tasks // 2)
                            res[i] = (
                                interval * (current_task_count_ - n_tasks // 2)
                                + dig_range
                            )
                        else:
                            interval = (-dig_range - low_) / (n_tasks // 2)
                            res[i] = (
                                interval * (n_tasks // 2 - current_task_count_) + low_
                            )
                    else:
                        while True:
                            rand = np.random.uniform(low_, up_)
                            if rand > dig_range or rand < -dig_range:
                                res[i] = rand
                                break
                res = np.array(res).reshape(size)
                return res

            uniform = uniform_function
        bound = lambda x: np.array(1.5) ** uniform(
            -self.log_scale_limit, self.log_scale_limit, x
        )
        bound_uniform = lambda x: uniform(
            -self.log_scale_limit, self.log_scale_limit, x
        )
        for _ in range(n_tasks):
            new_params = {}

            if "body_mass" in self.rand_params:
                body_mass_multipliers = bound((1,))
                new_params["body_mass"] = body_mass_multipliers

            if "dof_damping_1_dim" in self.rand_params:
                dof_damping_1_dim_multipliers = bound((1,))
                new_params["dof_damping_1_dim"] = dof_damping_1_dim_multipliers

            if "gravity" in self.rand_params:
                gravity_mutipliers = bound(self.model.opt.gravity.shape)
                new_params["gravity"] = np.multiply(
                    self.init_params["gravity"], gravity_mutipliers
                )

                if "gravity_angle" in self.rand_params:
                    min_angle = -self.log_scale_limit * np.array([1, 1]) / 8
                    max_angle = self.log_scale_limit * np.array([1, 1]) / 8
                    angle = np.random.uniform(min_angle, max_angle)
                    new_params["gravity"][0] = (
                        new_params["gravity"][2] * np.sin(angle[0]) * np.sin(angle[1])
                    )
                    new_params["gravity"][1] = (
                        new_params["gravity"][2] * np.sin(angle[0]) * np.cos(angle[1])
                    )
                    new_params["gravity"][2] *= np.cos(angle[0])

            param_sets.append(new_params)
            current_task_count_ += 1

        return param_sets

    def cross_params(self, param_a, param_b):
        param_res = []
        for item_a in param_a:
            for item_b in param_b:
                r = dict()
                for k, v in item_a.items():
                    r[k] = v
                for k, v in item_b.items():
                    r[k] = v
                param_res.append(r)
        return param_res

    def set_task(self, task):
        if self.is_diy_env:
            self.env.set_task(task)
            self.cur_parameter_vector = self.env_parameter_vector_
            return
        for param, param_val in task.items():
            if param == "gravity":
                param_variable = getattr(self.unwrapped.model.opt, param)
                param_variable[:] = param_val    
            elif param == "body_mass":
                param_variable = getattr(self.unwrapped.model, "body_mass")  
                # param_variable[:] = param_val
                param_variable[:] = self.init_params[param]
                param_variable[1] = self.init_params[param][1] * param_val 
                continue
            elif param == "dof_damping_1_dim":
                param_variable = getattr(self.unwrapped.model, "dof_damping")
                param_variable[:] = self.init_params[param][:] * param_val 
                continue
            else:
                param_variable = getattr(self.unwrapped.model, param)
                param_variable[:] = param_val
            assert (
                param_variable.shape == param_val.shape
            ), "shapes of new parameter value and old one must match"

        self.cur_params = task
        self.cur_parameter_vector = self.env_parameter_vector_

    def get_task(self):
        return self.cur_params

    def save_parameters(self):
        self.init_params = {}
        if "body_mass" in self.rand_params:
            self.init_params["body_mass"] = self.unwrapped.model.body_mass

        if "dof_damping_1_dim" in self.rand_params:
            self.init_params["dof_damping_1_dim"] = np.array(
                self.unwrapped.model.dof_damping
            ).copy()

        if "gravity" in self.rand_params:
            self.init_params["gravity"] = self.unwrapped.model.opt.gravity

        self.cur_params = copy.deepcopy(self.init_params)
        if "dof_damping_1_dim" in self.cur_params:
            self.cur_params["dof_damping_1_dim"] = np.array([1.0])

    @property
    def env_parameter_vector(self):
        return self.cur_parameter_vector

    @property
    def env_parameter_vector_(self):
        if self.is_diy_env:
            return self.env.env_parameter_vector_
        keys = [key for key in self.rand_params]
        if len(keys) == 0:
            return []
        vec_ = [self.cur_params[key].reshape(-1) for key in keys]
        cur_vec = np.hstack(vec_)
        if not self.normalize_context:
            return cur_vec
        vec_range = self.param_max - self.param_min
        vec_range[vec_range == 0] = 1.0

        cur_vec = (cur_vec - self.param_min) / vec_range
        return cur_vec

    @property
    def env_parameter_length(self):
        if self.is_diy_env:
            return self.env.env_parameter_length
        length = np.sum(
            [np.shape(self.cur_params[key].reshape(-1))[-1] for key in self.cur_params]
        )
        return length

    @property
    def param_max(self):
        keys = [key for key in self.rand_params]
        vec_ = [self.max_param[key].reshape(-1) for key in keys]
        if len(vec_) == 0:
            return []
        return np.hstack(vec_)

    @property
    def param_min(self):
        keys = [key for key in self.rand_params]
        vec_ = [self.min_param[key].reshape(-1) for key in keys]
        if len(vec_) == 0:
            return []
        return np.hstack(vec_)

    @property
    def _elapsed_steps(self):
        return self.cur_step_ind

    @property
    def _max_episode_steps(self):
        if hasattr(self.env, "_max_episode_steps"):
            return self.env._max_episode_steps
        return 1000
