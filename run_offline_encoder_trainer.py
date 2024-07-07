import os
import gym
from stable_baselines3.common.utils import set_random_seed

from trainer.encoder_trainer import Trainer
from envs.nonstationary_env import NonstationaryEnv
from utils.config import get_cfg, get_policy_config
from utils.env import make_env_param_dict_from_params, set_env_seed
from utils.logger import Logger
import numpy as np
import re
import json

### the path of your offline datasets###
DATASET_ROOT_DIR = ''

ENVS = {
    'cheetah-gravity' : 'HalfCheetah-v3',
    'hopper-gravity' : 'Hopper-v3',
    'walker-gravity' : 'Walker2d-v3',
    'cheetah-dof' : 'HalfCheetah-v3',
    'pendulum-gravity' : 'InvertedDoublePendulum-v2',
    'cheetah-body_mass' : 'HalfCheetah-v3'
}


PARAMS = {
    'cheetah-gravity' : 'gravity',
    'hopper-gravity' : 'gravity',
    'walker-gravity' : 'gravity',
    'cheetah-dof' : 'dof_damping_1_dim',
    'pendulum-gravity' : 'gravity',
    'cheetah-body_mass' : 'body_mass'
}



if __name__ == "__main__":
    # read config
    parameter = get_cfg()
    base_path = os.getcwd()
    set_random_seed(parameter.seed, True)
    # search dataset
    train_tasks_path = []
    datasets_path = []
    train_tasks = []

    env_type = parameter.env_type

    env_name = ENVS[env_type]
    params = [PARAMS[env_type]]
    parameter.dataset_path = os.path.join(DATASET_ROOT_DIR, env_type)

    for root, dirs, files in os.walk(parameter.dataset_path):
        for file in files:
            if '.hdf5' in file:
                datasets_path.append(os.path.join(root, file))
            if '.json' in file:
                train_tasks_path.append(os.path.join(root, file))
                
    findID = lambda path : int(re.search('task\d+', path).group(0)[4 : ])
    datasets_path.sort(key = findID)
    train_tasks_path.sort(key = findID)
    for train_task in train_tasks_path:
        with open(train_task, 'rb') as f:
            p = json.load(f)
        train_tasks.append(p)
    
    env_parameter_dict = make_env_param_dict_from_params(
        params, train_tasks
    )
    parameter.task_num = len(datasets_path)
    # logger
    logger = Logger(base_path, parameter, output_dir = 'log')
    logger(parameter)
    with open(os.path.join(logger.output_dir, 'offline.txt'), 'x') as f:
        f.write(f'dataset : {parameter.dataset_path}')
    f.close()

    # encoder config
    env = gym.make(env_name)
    policy_config = get_policy_config(
        parameter, env.observation_space.shape[0], env.action_space.shape[0]
    )
    
    trainer = Trainer(
        datasets = datasets_path,
        task_params = train_tasks_path,
        parameter = parameter,
        env_parameter_dict = env_parameter_dict,
        policy_config = policy_config,
        logger = logger
    )
    trainer.train()
