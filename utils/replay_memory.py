import random
from collections import namedtuple
from typing import List, NamedTuple

import numpy as np

tuplenames = (
    "state",
    "action",
    "mask",
    "next_state",
    "reward",
    "next_action",
    "task",
    "env_param",
    "last_action",
    "done",
    "valid",
)
Transition = namedtuple("Transition", tuplenames)
nd_type = NamedTuple(
    "Transition",
    [
        ("state", np.ndarray),
        ("action", np.ndarray),
        ("mask", float),
        ("next_state", np.ndarray),
        ("reward", float),
        ("next_action", np.ndarray),
        ("task", int),
        ("env_param", List[float]),
        ("last_action", np.ndarray),
        ("done", bool),
        ("valid", bool),
    ],
)


class Memory:
    def __init__(self):
        self.memory: List[nd_type] = []
        self.max_size = -1

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int = None):
        if batch_size is None or batch_size > len(self.memory):
            return Transition(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return Transition(*zip(*random_batch))

    def append(self, new_memory: List[nd_type]):
        self.memory += new_memory

    def __len__(self):
        return len(self.memory)

    @property
    def size(self):
        return len(self.memory)


class MemoryArray(object):
    def __init__(
        self,
        rnn_slice_length: int = 32,
        max_trajectory_num: int = 1000,
        max_traj_step: int = 1050,
        fix_length: int = 0,
    ):
        self.memory = []
        self.trajectory_length = [0] * max_trajectory_num
        self.available_traj_num = 0
        self.memory_buffer = None
        self.ind_range = None
        self.ptr = 0
        self.max_trajectory_num = max_trajectory_num
        self.max_traj_step = max_traj_step
        self.fix_length = fix_length
        self.transition_buffer = []
        self.transition_count = 0
        self.rnn_slice_length = rnn_slice_length
        self._last_saving_time = 0
        self._last_saving_size = 0
        self.last_sampled_batch = None

    @staticmethod
    def get_max_len(max_len: int, slice_length: int):
        if max_len % slice_length == 0 and max_len > 0:
            return max_len
        else:
            max_len = (max_len // slice_length + 1) * slice_length
        return max_len

    def sample_fix_length_sub_trajs(self, batch_size: int, fix_length: int):
        list_ind = np.random.randint(0, self.transition_count, (batch_size))
        res = [self.transition_buffer[ind] for ind in list_ind]
        if (
            self.last_sampled_batch is None
            or not self.last_sampled_batch.shape[0] == batch_size
            or not self.last_sampled_batch.shape[1] == fix_length
        ):
            trajs = [
                self.memory_buffer[traj_ind, point_ind + 1 - fix_length : point_ind + 1]
                for traj_ind, point_ind in res
            ]
            trajs = np.array(trajs, copy=True)
            self.last_sampled_batch = trajs
        else:
            for ind, (traj_ind, point_ind) in enumerate(res):
                self.last_sampled_batch[ind, :, :] = self.memory_buffer[
                    traj_ind, point_ind + 1 - fix_length : point_ind + 1, :
                ]

        res = self.array_to_transition(self.last_sampled_batch)
        return res

    def sample_trajs(self, batch_size: int, max_sample_size: int = None):
        mean_traj_len = self.transition_count / self.available_traj_num
        desired_traj_num = max(int(batch_size / mean_traj_len), 1)
        if max_sample_size is not None:
            max_traj_num = max_sample_size // self.max_traj_step
            desired_traj_num = min(desired_traj_num, max_traj_num)
        traj_inds = np.random.randint(
            0, self.available_traj_num, (int(desired_traj_num))
        )
        trajs = self.memory_buffer[traj_inds]
        traj_len = [self.trajectory_length[ind] for ind in traj_inds]
        max_traj_len = max(traj_len)
        max_traj_len = self.get_max_len(max_traj_len, self.rnn_slice_length)
        trajs = trajs[:, :max_traj_len, :]
        total_size = sum(traj_len)

        return self.array_to_transition(trajs), total_size

    def transition_to_array(self, transition: nd_type):
        res = []
        for item in transition:
            if item is not None:
                if isinstance(item, np.ndarray):
                    res.append(item.reshape((1, -1)))
                elif isinstance(item, list):
                    res.append(np.array(item).reshape((1, -1)))
                else:
                    raise NotImplementedError(
                        "not implement for type of {}".format(type(item))
                    )
        res = np.hstack(res)
        assert (
            res.shape[-1] == self.memory_buffer.shape[-1]
        ), "data_size: {}, buffer_size: {}".format(res.shape, self.memory_buffer.shape)
        return np.hstack(res)

    def array_to_transition(self, data: List):
        data_list = []
        for item in self.ind_range:
            if len(item) > 0:
                start = item[0]
                end = item[-1] + 1
                data_list.append(data[..., start:end])
            else:
                data_list.append(None)
        res = Transition(*data_list)
        return res

    def complete_traj(self, memory: Memory):
        if self.memory_buffer is None:
            start_dim = 0
            self.ind_range = []
            self.trajectory_length = [0] * self.max_trajectory_num
            end_dim = 0
            for item in memory[0]:
                dim = 0
                if type(item) is np.ndarray:
                    dim = item.shape[-1]
                elif isinstance(item, list):
                    dim = len(item)
                elif item is None:
                    dim = 0
                end_dim = start_dim + dim
                self.ind_range.append(list(range(start_dim, end_dim)))
                start_dim = end_dim
            self.memory_buffer = np.zeros(
                (self.max_trajectory_num, self.max_traj_step + self.fix_length, end_dim)
            )
        for ind, transition in enumerate(memory):
            self.memory_buffer[
                self.ptr, ind + self.fix_length, :
            ] = self.transition_to_array(transition)
            self.transition_buffer.append((self.ptr, ind + self.fix_length))
        self.transition_count -= self.trajectory_length[self.ptr]
        if self.trajectory_length[self.ptr] > 0:
            self.transition_buffer[: self.trajectory_length[self.ptr]] = []
        self.trajectory_length[self.ptr] = len(memory)

        self.ptr = (self.ptr + 1) % self.max_trajectory_num
        self.available_traj_num = max(self.available_traj_num, self.ptr)
        self.transition_count += len(memory)

    def mem_push_array(self, mem: Memory):
        for item in mem.memory:
            self.memory += [item]
            if item.done[0]:
                self.complete_traj(self.memory)
                self.memory = []

    def __len__(self):
        return self.available_traj_num

    @property
    def size(self):
        return self.transition_count