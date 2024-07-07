from gym import Wrapper, Env
from typing import Dict
import numpy as np
import json

class ParaWrapper(Wrapper):
    RAND_PARAMS = ["body_mass", "dof_damping_1_dim", "dof_damping" "gravity"]
    RAND_PARAMS_EXTENDED = RAND_PARAMS

    def __init__(self,
                 env: Env,
                ) -> None:
        super().__init__(env)
        self.init_params = {}
        self._get_init_params()

    def _get_init_params(self):
        
        self.init_params['body_mass'] = self.model.body_mass
        self.init_params['gravity'] = self.model.opt.gravity
        self.init_params['dof_damping_1_dim'] = np.array(self.model.dof_damping).copy()
        self.init_params['dof_damping'] = np.array(self.model.dof_damping).copy()
        
    @staticmethod
    def load_json(path : str):
        with open(path, 'r') as f:
            para = json.load(f)
        f.close()

        return para
    
    def set_para(self, task_parameters : Dict):

        for param, param_val in task_parameters.items():
            if param == "body_mass":
                self.model.body_mass[:][1] = self.init_params['body_mass'][1] * param_val[0]
            elif param == "dof_damping":
                self.model.dof_damping[:] = param_val
            elif param == "gravity":
                self.model.opt.gravity[:] = param_val
            elif param == 'dof_damping_1_dim':
                self.model.dof_damping[ : ] = self.init_params['dof_damping_1_dim'] * param_val
            else:
                raise NotImplementedError(
                    f"Parameter {param} is not supported to modified."
                )
        self.param_vec = np.array(param_val)
