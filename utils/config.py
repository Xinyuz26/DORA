import argparse
from typing import NamedTuple


def get_value_config(parameter: NamedTuple, obs_dim: int, act_dim: int):
    return dict(
        obs_dim=obs_dim,
        act_dim=act_dim,
        up_hidden_size=parameter.value_hidden_size,
        up_activations=parameter.value_activations,
        up_layer_type=parameter.value_layer_type,
        ep_hidden_size=parameter.ep_hidden_size,
        ep_activation=parameter.ep_activations,
        ep_layer_type=parameter.ep_layer_type,
        ep_dim=parameter.ep_dim,
        use_gt_env_feature=parameter.use_true_parameter,
    )


def get_policy_config(parameter: NamedTuple, obs_dim: int, act_dim: int):
    return dict(
        obs_dim=obs_dim,
        act_dim=act_dim,
        up_hidden_size=parameter.up_hidden_size,
        up_activations=parameter.up_activations,
        up_layer_type=parameter.up_layer_type,
        ep_hidden_size=parameter.ep_hidden_size,
        ep_activation=parameter.ep_activations,
        ep_layer_type=parameter.ep_layer_type,
        ep_dim=parameter.ep_dim,
        use_gt_env_feature=parameter.use_true_parameter,
        rnn_fix_length=parameter.rnn_fix_length,
        bottle_sigma=parameter.bottle_sigma,
    )


def get_cfg():
    parser = argparse.ArgumentParser()

    parser.add_argument("--description", type=str, default="DEBUG")

    parser.add_argument(
        "--env_type",
        type=str,
        default="HalfCheetah-v3",
        help="name of the environment to run",
    )
    
    parser.add_argument(
        "--dataset_path", type=str, default=""
    )

    parser.add_argument(
        "--model_path", type=str, default="", help="path of pre-trained model"
    )
    parser.add_argument(
        "--render", action="store_true", default=False, help="render the environment"
    )
    parser.add_argument(
        "--log_std", type=float, default=-1.5, help="log std for the policy"
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument(
        "--policy_learning_rate", type=float, default=3e-4, help="learning rate"
    )
    parser.add_argument(
        "--value_learning_rate", type=float, default=1e-3, help="learning rate"
    )
    parser.add_argument(
        "--num_threads", type=int, default=4, help="number of threads for agent"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    parser.add_argument(
        "--random_num",
        type=int,
        default=4000,
        help="sample random_num fully random samples,",
    )

    parser.add_argument(
        "--non_stationary_period",
        type=int,
        default=100,
        help="How many steps the env changes",
    )

    parser.add_argument("--non_stationary_interval", type=int, default=10)

    parser.add_argument(
        "--start_train_num",
        type=int,
        default=20000,
        help="after reach start_train_num, training start",
    )

    parser.add_argument(
        "--test_sample_num", type=int, default=4000, help="sample num in test phase"
    )

    parser.add_argument(
        "--sac_update_time",
        type=int,
        default=1000,
        help="update time after sampling a batch data",
    )

    parser.add_argument(
        "--sac_replay_size",
        type=int,
        default=1e6,
        help="update time after sampling a batch data",
    )

    parser.add_argument(
        "--min_batch_size",
        type=int,
        default=1200,
        help="minimal sample number per iteration",
    )

    parser.add_argument(
        "--sac_mini_batch_size",
        type=int,
        default=256,
        help="sac_mini_batch_size trajectories will be sampled from the replay buffer.",
    )

    parser.add_argument(
        "--sac_inner_iter_num",
        type=int,
        default=1,
        help="after sample several trajectories from replay buffer, "
        "sac_inner_iter_num mini-batch will be sampled from the batch, "
        "and model will be optimized for sac_inner_iter_num times.",
    )

    parser.add_argument(
        "--rnn_sample_max_batch_size",
        type=int,
        default=3e5,
        help="max point num sampled from replay buffer per time",
    )

    parser.add_argument(
        "--rnn_slice_num", type=int, default=16, help="gradient clip steps"
    )

    parser.add_argument(
        "--sac_tau",
        type=float,
        default=0.995,
        help="ratio of coping value net to target value net",
    )

    parser.add_argument(
        "--sac_alpha", type=float, default=0.2, help="sac temperature coefficient"
    )

    parser.add_argument(
        "--max_iter_num",
        type=int,
        default=10000,
        help="maximal number of main iterations (default: 500)",
    )

    parser.add_argument(
        "--save_model_interval",
        type=int,
        default=5,
        help="interval between saving model (default: 5, means don't save)",
    )

    parser.add_argument("--update_interval", type=int, default=1)

    parser.add_argument(
        "--target_entropy_ratio", type=float, default=1.5, help="target entropy"
    )

    parser.add_argument("--history_length", type=int, default=0)

    parser.add_argument("--task_num", type=int, default=0)

    parser.add_argument(
        "--test_task_num", type=int, default=0, help="number of tasks for testing"
    )

    parser.add_argument("--use_true_parameter", action="store_true", default=False)
    parser.add_argument(
        "--bottle_sigma",
        type=float,
        default=1e-2,
        help="std of the noise injected to ep while inference (information bottleneck)",
    )
    parser.add_argument(
        "--l2_norm_for_ep", type=float, default=0.0, help="L2 norm added to EP module"
    )

    parser.add_argument(
        "--rnn_fix_length",
        type=int,
        default=8,
        help="fix the rnn memory length to rnn_fix_length",
    )
    parser.add_argument(
        "--minimal_repre_rp_size",
        type=float,
        default=1e5,
        help="after minimal_repre_rp_size, start training EP module",
    )

    parser.add_argument(
        "--ep_start_num",
        type=int,
        default=0,
        help="only when the size of the replay buffer is larger than ep_start_num"
        ", ep can be learned",
    )
    parser.add_argument(
        "--kernel_type",
        default="rbf_element_wise",
        help="kernel type for DPP loss computing (rbf/rbf_element_wise/inner)",
    )

    parser.add_argument(
        "--rmdm_ratio", type=float, default=1.0, help="gradient ratio of rmdm"
    )

    parser.add_argument(
        "--rmdm_tau",
        type=float,
        default=0.995,
        help="smoothing ratio of the representation",
    )

    parser.add_argument(
        "--repre_loss_factor",
        type=float,
        default=1.0,
        help="size of the representation loss",
    )
    parser.add_argument(
        "--ep_smooth_factor",
        type=float,
        default=0.0,
        help="smooth  factor for ep module, 0.0 for apply concurrently",
    )

    parser.add_argument(
        "--rbf_radius", type=float, default=80.0, help="radius of the rbf kerel"
    )

    parser.add_argument(
        "--env_default_change_range",
        type=float,
        default=1.0,
        help="environment default change range",
    )

    parser.add_argument(
        "--env_ood_change_range",
        type=float,
        default=2.0,
        help="environment OOD change range",
    )

    parser.add_argument(
        "--consistency_loss_weight",
        type=float,
        default=50.0,
        help="If you want to change the ratio of distortion loss and debias loss, you need to change w_1 and w_2 in encoder_trainer.py",
    )

    parser.add_argument(
        "--diversity_loss_weight",
        type=float,
        default=0.025,
        help="If you want to change the ratio of distortion loss and debias loss, you need to change w_1 and w_2 in encoder_trainer.py",
    )

    parser.add_argument(
        "--varying_params", nargs="+", type=str, default=["gravity"]
    )

    parser.add_argument(
        "--up_hidden_size",
        nargs="+",
        type=int,
        default=[128, 64],
        help="architecture of the hidden layers of Universe Policy",
    )

    parser.add_argument(
        "--up_activations",
        nargs="+",
        type=str,
        default=["leaky_relu", "leaky_relu", "linear"],
        help="activation of each layer of Universe Policy",
    )

    parser.add_argument(
        "--up_layer_type",
        nargs="+",
        type=str,
        default=["fc", "fc", "fc"],
        help="net type of Universe Policy",
    )

    parser.add_argument(
        "--ep_hidden_size",
        nargs="+",
        type=int,
        default=[128, 64],
        help="architecture of the hidden layers of Environment Probing Net",
    )

    parser.add_argument(
        "--ep_activations",
        nargs="+",
        type=str,
        default=["leaky_relu", "linear", "tanh"],
        help="activation of each layer of Environment Probing Net",
    )

    parser.add_argument(
        "--ep_layer_type",
        nargs="+",
        type=str,
        default=["fc", "gru", "fc"],
        help="net type of Environment Probing Net",
    )

    parser.add_argument(
        "--ep_dim", type=int, default=2, help="dimension of environment features"
    )

    parser.add_argument(
        "--value_hidden_size",
        nargs="+",
        type=int,
        default=[128, 64],
        help="architecture of the hidden layers of value",
    )

    parser.add_argument(
        "--value_activations",
        nargs="+",
        type=str,
        default=["leaky_relu", "leaky_relu", "linear"],
        help="activation of each layer of value",
    )

    parser.add_argument(
        "--value_layer_type",
        nargs="+",
        type=str,
        default=["fc", "fc", "fc"],
        help="net type of value",
    )

    parser.add_argument(
        '--stop_update',
        type = float,
        default = 0.1
    )

    return parser.parse_args()
