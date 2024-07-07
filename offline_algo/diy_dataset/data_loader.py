import h5py
from tqdm import tqdm
import numpy as np

def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys

def data_loader(h5path):
    data_dict = {}
    with h5py.File(h5path, 'r') as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            # print(k)
            d4rl_k = k
            if k == "dones":
                d4rl_k = "terminals"
            
            try:  # first try loading as an array
                data_dict[d4rl_k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict[d4rl_k] = dataset_file[k][()]
            
            if k == "rewards":
                print("dataset length:")
                print(len(dataset_file[k]))
                print("last trajectory rewards")
                print(np.sum(dataset_file[k][-1000:]))

    return data_dict

def data_loader_return_with_path(h5path):
    data_dict = {}
    with h5py.File(h5path, 'r') as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            # print(k)
            d4rl_k = k
            if k == "dones":
                d4rl_k = "terminals"
            
            try:  # first try loading as an array
                data_dict[d4rl_k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict[d4rl_k] = dataset_file[k][()]
            
            if k == "rewards":
                print("dataset length:")
                print(len(dataset_file[k]))
                print("last trajectory rewards")
                last_trajectory_reward = np.sum(dataset_file[k][-1000:])
                print(last_trajectory_reward)

    return data_dict, last_trajectory_reward




def keep_part_dataset(h5path):
    data_dict = {}
    new_h5path =  h5path
    with h5py.File(h5path, 'r') as dataset_file:
        total_transitions = len(dataset_file['rewards'])
        num_transitions_to_keep = 200000

        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            d4rl_k = k
            # if k == "dones":
            #     d4rl_k = "terminals"

            data_dict[d4rl_k] = dataset_file[k][:num_transitions_to_keep]

            if k == "rewards":
                print("dataset length:")
                print(len(data_dict[k]))
                print("last trajectory rewards")
                last_trajectory_reward = np.sum(data_dict[k][-1000:])
                print(last_trajectory_reward)

    # Save the modified data back to the original file
    with h5py.File(new_h5path, 'w') as new_dataset_file:
        for k, v in data_dict.items():
            new_dataset_file.create_dataset(k, data=v)

    return data_dict, last_trajectory_reward