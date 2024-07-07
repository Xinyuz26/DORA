import sys
sys.path.append('.')
import h5py
import os
import json
import re
import numpy as np
from typing import Dict, List

def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys

def data_loader(h5path : str):
    data_dict = {}
    with h5py.File(h5path, 'r') as dataset_file:
        # for k in tqdm(get_keys(dataset_file), desc="load datafile"):
        for k in get_keys(dataset_file):
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]

    return data_dict

def json_to_vec(json_file_path : str):
    assert os.path.exists(json_file_path)
    with open(json_file_path, 'r') as f:
        p = json.load(f)
    f.close()
    vec = []
    for val in p.values():
        vec.append(val)
    vec = np.array(vec).reshape((-1, ))
    
    return vec

def split_dataset_into_trajs(
    dataset: Dict[str, np.ndarray], max_episode_steps: int = 1000
):
    """Split the [D4RL] style dataset into trajectories
    
    :return: the corresponding start index and end index (not included) of every trajectories
    """
    max_steps = dataset["observations"].shape[0]
    if "timeouts" in dataset:
        timeout_idx = np.where(dataset["timeouts"] == True)[0] + 1
        terminal_idx = np.where(dataset["dones"] == True)[0] + 1
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
                    if dataset["dones"][i - 1]:
                        break
                traj_pairs.append([start_idx, i])

    return traj_pairs

def json_loader(json_file_path : str):
    assert os.path.exists(json_file_path)
    with open(json_file_path, 'r') as f:
        json_str = f.read()
    p = json.loads(json_str)
    f.close()
    for key in p.keys():
        find_floats = re.findall(r"[-+]?\d*\.\d+e?[-+]?\d*", p[key])
        if len(find_floats) == 1:
            p[key] = float(find_floats[0])
        else:
            p[key] = np.array([float(num) for num in find_floats])
                 
    return p