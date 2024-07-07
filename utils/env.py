import math
from typing import Dict, List, Tuple, Union
from gym import Env

def set_env_seed(env: Union[Env, Tuple[Env, ...]], seed: int = 0):
    def _set_env_seed(env: Env, seed: int):
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

    if type(env) is tuple:
        for e in env:
            _set_env_seed(e, seed)
    else:
        _set_env_seed(env, seed)


def make_env_param_dict_from_params(params: List[str], env_tasks: List) -> Dict:
    def _make_env_param_dict(parameter_name: str):
        res = {}
        if env_tasks is not None:
            for ind, item in enumerate(env_tasks):
                res[ind + 1] = item
        res_interprete = {}
        for k, v in res.items():
            if isinstance(v, dict):
                res_interprete[k] = [v[parameter_name][-1]]
            elif isinstance(v, int):
                res_interprete[k] = v
            elif isinstance(v, list):
                res_interprete[k] = math.sqrt(sum([item ** 2 for item in v]))
            else:
                raise NotImplementedError(f"type({type(v)}) is not implemented.")
        return res_interprete

    res_interprete = {}
    for param in params:
        res_ = _make_env_param_dict(param)
        for k, v in res_.items():
            if k not in res_interprete:
                res_interprete[k] = v
            else:
                res_interprete[k] += v

    return res_interprete
