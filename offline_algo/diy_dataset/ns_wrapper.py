from gym import Wrapper, Env
from typing import Dict, List, Union
import numpy as np

class NS_Wrapper(Wrapper):
    def __init__(self, 
                 env: Env,
                 change_interval : int = 100,
                ):
        
        super().__init__(env)
        self.change_interval = change_interval

        self.timestep = None        
        self.ns_params = None
        self.param_index = None
        self.allow_param_types = ['gravity', 'dof_damping_1_dim', 'body_mass']
        self.init_params = {}
        self._get_init_params()

    def _get_init_params(self):
        self.init_params['body_mass'] = self.model.body_mass
        self.init_params['gravity'] = self.model.opt.gravity
        self.init_params['dof_damping_1_dim'] = np.array(self.model.dof_damping).copy()

    def _set_para(self, task_parameters: Dict):
        
        for param, param_val in task_parameters.items():
            if param == "body_mass":
                self.model.body_mass[:][1] = self.init_params['body_mass'][1] * param_val[0]
            elif param == "gravity":
                self.model.opt.gravity[:] = param_val
            elif param == 'dof_damping_1_dim':
                self.model.dof_damping[ : ] = self.init_params['dof_damping_1_dim'] * param_val
            else:
                raise NotImplementedError(
                    f"Parameter {param} is not supported to modified."
                )
    
    def sample_tasks(self, params_types : Union[str, List[str]], nums : int = 10, scale : float = 1.5):
        task_cnt = 0
        def uniform_fn(low, up, size):
            res = [0] * np.prod(size)
            interval = (up - low) / (nums - 1)
            for i in range(len(res)):
                res[i] = interval * task_cnt + low
            
            return res
        
        bound = lambda x : np.array(1.5) ** uniform_fn(-scale, scale, x)
        params_types = params_types if isinstance(params_types, List) else [params_types]
        
        tasks = []
        while task_cnt < nums:
            task_param = {}
            if 'body_mass' in params_types:
                body_mass_multipliers = bound((1,))
                task_param["body_mass"] = body_mass_multipliers
            
            if 'dof_damping_1_dim' in params_types:
                dof_damping_1_dim_multipliers = bound((1,))
                task_param["dof_damping_1_dim"] = dof_damping_1_dim_multipliers
            
            if 'gravity' in params_types:
                gravity_mutipliers = bound(self.model.opt.gravity.shape)
                task_param["gravity"] = np.multiply(
                    self.init_params['gravity'], gravity_mutipliers
                )
            
            task_cnt += 1
            tasks.append(task_param)
        
        return tasks
    
    def sample_pure_ood_tasks(self, params_types : Union[str, List[str]], nums : int = 10, iid_scale : float = 1.5, ood_scale : float = 1.8):
        task_cnt = 0
        assert nums % 2 == 0
        def uniform_fn(low, up, size, point_nums : int = nums // 2):
            res = [0] * np.prod(size)
            interval = (up - low) / (point_nums - 1)
            for i in range(len(res)):
                res[i] = interval * task_cnt + low
            
            return res
        
        bound_1 = lambda x : np.array(1.5) ** uniform_fn(-ood_scale, -iid_scale, x)
        bound_2 = lambda x : np.array(1.5) ** uniform_fn(iid_scale, ood_scale, x)
        params_types = params_types if isinstance(params_types, List) else [params_types]
        
        tasks = []
        while task_cnt < nums // 2:
            task_param_lower = {}
            task_param_upper = {}
            if 'body_mass' in params_types:
                body_mass_multipliers_lower = bound_1((1,))
                body_mass_multipliers_upper = bound_2((1,))
                task_param_lower["body_mass"] = body_mass_multipliers_lower
                task_param_upper["body_mass"] = body_mass_multipliers_upper
            
            if 'dof_damping_1_dim' in params_types:
                dof_damping_1_dim_multipliers_lower = bound_1((1,))
                dof_damping_1_dim_multipliers_upper = bound_2((1, ))
                task_param_lower["dof_damping_1_dim"] = dof_damping_1_dim_multipliers_lower
                task_param_upper['dof_damping_1_dim'] = dof_damping_1_dim_multipliers_upper
            
            if 'gravity' in params_types:
                gravity_mutipliers_lower = bound_1(self.model.opt.gravity.shape)
                gravity_mutipliers_upper = bound_2(self.model.opt.gravity.shape)
                task_param_lower["gravity"] = np.multiply(
                    self.init_params['gravity'], gravity_mutipliers_lower
                )
                task_param_upper["gravity"] = np.multiply(
                    self.init_params['gravity'], gravity_mutipliers_upper
                )
            
            task_cnt += 1
            tasks.append(task_param_lower)
            tasks.append(task_param_upper)
        
        return tasks
    
    def set_ns_params(self, task_params : List[Dict]):
        self.ns_params = task_params
    
    def _get_ns_param_scalar(self):
        param = self.ns_params[self.param_index]
        key = list(param.keys())[-1]
        value = param[key]
        if not isinstance(value, int) and not isinstance(value, float):
            value = value[-1]
        
        return value
    
    def reset(self):
        assert self.ns_params != None
        self.timestep = 0
        self.param_index = np.random.choice(len(self.ns_params))
        self._set_para(self.ns_params[self.param_index])

        state = self.env.reset()
        return state
    
    def step(self, action):
        assert self.timestep != None

        if self.timestep > 0 and self.timestep % self.change_interval == 0:
            self.param_index = np.random.choice(len(self.ns_params))
            self._set_para(self.ns_params[self.param_index])
        
        next_obs, reward, terminal, info = self.env.step(action)
        info['ns_scalar'] = self._get_ns_param_scalar()
        self.timestep += 1

        return next_obs, reward, terminal, info