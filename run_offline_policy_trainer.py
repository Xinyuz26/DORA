import argparse
import random
import gym
import numpy as np
import torch
import os
from offline_algo.nets import MLP
from offline_algo.modules import ActorProb, Critic, TanhDiagGaussian
from offline_algo.utils.load_dataset import qlearning_dataset
from offline_algo.buffer import ReplayBuffer, OfflineMultiTaskBuffer
from offline_algo.utils.logger import Logger, make_log_dirs
from offline_algo.policy_trainer import MetaMFPolicyTrainer
from offline_algo.policy import MetaCQLPolicy
from offline_algo.diy_dataset.data_loader import data_loader
from offline_algo.diy_dataset.rand_wrapper import ParaWrapper

### the path of your offline datasets###
ROOT_DATASET_DIR = ''

ENVS = {
    'cheetah' : 'HalfCheetah-v3',
    'hopper' : 'Hopper-v3',
    'walker' : 'Walker2d-v3',
    'pendulum' : 'InvertedDoublePendulum-v2',
    'cheetah-body_mass' : 'HalfCheetah-v3'
}
DATASETS = ['cheetah-gravity', 'cheetah-dof', 
            'hopper-gravity', 
            'walker-gravity',
            'pendulum-gravity',
            'cheetah-body_mass'
        ]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="cql")
    parser.add_argument("--env", type=str, default="cheetah-gravity")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256])
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--target-entropy", type=int, default=None)
    parser.add_argument("--auto-alpha", default=True)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)

    parser.add_argument("--cql-weight", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-q-backup", type=bool, default=False)
    parser.add_argument("--deterministic-backup", type=bool, default=True)
    parser.add_argument("--with-lagrange", type=bool, default=False)
    parser.add_argument("--lagrange-threshold", type=float, default=10.0)
    parser.add_argument("--cql-alpha-lr", type=float, default=3e-4)
    parser.add_argument("--num-repeat-actions", type=int, default=10)
    
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # new env
    parser.add_argument("--z_dim", type=int, default = 2)
    parser.add_argument("--rnn_fix_length", type=int, default = 8)
    parser.add_argument('--encoder_path', type = str, default = '')
    parser.add_argument('--prefix', type = str, default = '')
    
    return parser.parse_args()


def train(args=get_args()):
    # diy dataset
    assert args.env in DATASETS
    root_dataset_path = os.path.join(ROOT_DATASET_DIR, args.env)
    dataset_paths = []
    task_params_paths = []
    for root, dirs, files in os.walk(root_dataset_path):
        for file in files:
            if '.hdf5' in file:
                dataset_paths.append(os.path.join(root, file))
            if '.json' in file:
                task_params_paths.append(os.path.join(root, file))
    # sort by task id
    import re
    findID = lambda str : int(re.search('task\d+', str).group(0)[4 : ])
    dataset_paths.sort(key = findID)
    task_params_paths.sort(key = findID)
    datasets = []
    task_params = []
    for dataset_path in dataset_paths:
        datasets.append(data_loader(dataset_path))
    for task_param_path in task_params_paths:
        task_params.append(ParaWrapper.load_json(task_param_path))
        
    key = list(task_params[-1].keys())[-1]
    task_params_values = [task_param[key] for task_param in task_params]
    # create env
    env_id, para_type = args.env.split('-')
    env_id = ENVS[env_id]
    env = gym.make(env_id)
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)
    args.max_action = env.action_space.high[0]

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(args.seed)

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.z_dim, hidden_dims=args.hidden_dims)
    critic1_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.z_dim + args.action_dim , hidden_dims=args.hidden_dims)
    critic2_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.z_dim + args.action_dim , hidden_dims=args.hidden_dims)
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True
    )
    actor = ActorProb(actor_backbone, dist, args.device)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(env.action_space.shape)

        args.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha

    # create policy
    policy = MetaCQLPolicy(
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        action_space=env.action_space,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        cql_weight=args.cql_weight,
        temperature=args.temperature,
        max_q_backup=args.max_q_backup,
        deterministic_backup=args.deterministic_backup,
        with_lagrange=args.with_lagrange,
        lagrange_threshold=args.lagrange_threshold,
        cql_alpha_lr=args.cql_alpha_lr,
        num_repeart_actions=args.num_repeat_actions
    )

    buffer = OfflineMultiTaskBuffer(
        datasets, device = args.device, task_params = task_params_values
    )
    # log
    prefix = args.prefix
    if prefix != '':
        prefix += '_'
    log_dirs = make_log_dirs(args.env, prefix + args.algo_name, args.seed, vars(args))
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))
    
    # record source data path
    with open(os.path.join(log_dirs, 'source_dataset_path.txt'), 'x') as f:
        f.write(f'dataset : {root_dataset_path}')
    f.close()

    # create policy trainer
    policy_trainer = MetaMFPolicyTrainer(
        policy=policy,
        eval_env = env,
        buffer=buffer,
        logger=logger,
        obs_dim = np.prod(args.obs_shape),
        act_dim = int(args.action_dim),
        rnn_fix_length = args.rnn_fix_length,
        encoder_path= args.encoder_path,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        eval_episodes=args.eval_episodes,
        task_params = task_params,
    )

    # train
    use_ns_eval = False
    policy_trainer.set_use_ns_eval(use_ns_eval)
    policy_trainer.train()


if __name__ == "__main__":
    train()