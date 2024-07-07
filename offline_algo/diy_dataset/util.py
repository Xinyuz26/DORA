import numpy as np
from typing import Dict, List

def split_dataset_into_trajs(
    dataset: Dict[str, np.ndarray], max_episode_steps: int = 1000
):
    """Split the [D4RL] style dataset into trajectories
    
    :return: the corresponding start index and end index (not included) of every trajectories
    """
    max_steps = dataset["observations"].shape[0]
    terminal_key = 'terminals' if 'terminals' in list(dataset.keys()) else 'dones'
    if "timeouts" in dataset:
        timeout_idx = np.where(dataset["timeouts"] == True)[0] + 1
        terminal_idx = np.where(dataset[terminal_key] == True)[0] + 1
        start_idx = sorted(
            set(
                [0]
                + timeout_idx[timeout_idx < max_steps].tolist()
                + terminal_idx[terminal_idx < max_steps].tolist()
                + [max_steps]
            )
        )
        traj_pairs = list(zip(start_idx[:-1], start_idx[1:]))
    else:
        if max_episode_steps is None:
            raise Exception(
                "You have the specify the max_episode_steps if no timeouts in dataset"
            )
        else:
            traj_pairs = []
            i = 0
            while i < max_steps:
                start_idx = i
                traj_len = 1
                while (traj_len <= max_episode_steps) and (i < max_steps):
                    i += 1
                    traj_len += 1
                    if dataset[terminal_key][i - 1]:
                        break
                traj_pairs.append([start_idx, i])

    return traj_pairs

## The max scores and min scores of our test offline datasets, these scores are used to calculate the final test scores of algorithms.
## Thus you need to replace these scores when you test on new datasets
MAX_SCORES = {
    'cheetah-geom_size': 10064.611,
    'cheetah-dof': 8703.545,
    'cheetah-body_mass': 10060.59,
    'cheetah-gravity': 9218.485,
    'hopper-gravity': 3536.9714,
    'pendulum-gravity': 9351.943,
    'walker-gravity': 4630.102,
}
MIN_SCORES = {
    'cheetah-geom_size': -661.7589,
    'cheetah-dof': -613.0863,
    'cheetah-body_mass': -479.8587,
    'cheetah-gravity': -460.33096,
    'hopper-gravity': -1.220141,
    'pendulum-gravity': 26.705463,
    'walker-gravity': -110.49553,
}

def get_normalize_score(env_type, scores : List):
    max_score, min_score = MAX_SCORES[env_type], MIN_SCORES[env_type]
    rets = []
    for score in scores:
        rets.append(100 * (score - min_score) / (max_score - min_score))
    
    return rets