import copy
from typing import List
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_rbf_matrix(
    data: torch.Tensor,
    centers: torch.Tensor,
    alpha: float,
    element_wise_exp: bool = False,
):
    out_shape = torch.Size([data.shape[0], centers.shape[0], data.shape[-1]])
    data = data.unsqueeze(1).expand(out_shape)
    centers = centers.unsqueeze(0).expand(out_shape)
    if element_wise_exp:
        mtx = (-(centers - data).pow(2) * alpha).exp().mean(dim=-1, keepdim=False)
    else:
        mtx = (-(centers - data).pow(2) * alpha).sum(dim=-1, keepdim=False).exp()
    return mtx


def get_loss_dpp(y: torch.Tensor, kernel: str = "rbf", rbf_radius: float = 3000.0):
    if kernel == "rbf":
        K = (
            get_rbf_matrix(y, y, alpha=rbf_radius, element_wise_exp=False)
            + torch.eye(y.shape[0], device=y.device) * 1e-3
        )
    elif kernel == "rbf_element_wise":
        K = (
            get_rbf_matrix(y, y, alpha=rbf_radius, element_wise_exp=True)
            + torch.eye(y.shape[0], device=y.device) * 1e-3
        )
    elif kernel == "inner":
        K = y.matmul(y.t()).exp()
        K = K + torch.eye(y.shape[0], device=y.device) * 1e-3
        print(K)
    else:
        assert False
    loss = -torch.logdet(K)
    return loss


class RMDMLoss:
    def __init__(
        self,
        model_distance_matrix : np.ndarray,
        tau: float = 0.995,
        target_consistency_metric: float = -4.0,
        target_diverse_metric: torch.Tensor = None,
        max_env_len: int = 40,
        stop_update : float = 0.05
    ):
        self.mean_vector = {}
        self.tau = tau
        self.model_distance = torch.zeros(max_env_len, max_env_len).detach().cuda()
        self.aux_model_distance = self.model_distance + torch.eye(self.model_distance.shape[0]).cuda()
        self.target_consistency_metric = target_consistency_metric
        self.target_diverse_metric = target_diverse_metric
        self.lst_tasks = []
        self.max_env_len = max_env_len
        self.current_env_mean = None
        self.history_env_mean = None
        self.aux_kernel_radius = None  

        self.t = 0.0
        self.T = 50000.0
        self.tracing_consis_loss = deque(maxlen = 1000)
        self.allow_update = True
        self.anomaly_detection = False
        self.abnormal_batch = None
        self.stop_update = stop_update
        

    def construct_loss(
        self,
        consistency_loss: torch.Tensor,
        diverse_loss: torch.Tensor,
        consis_w: torch.Tensor,
        diverse_w: torch.Tensor,
        std: float,
    ):
        consis_w_loss = None
        divers_w_loss = None
        if isinstance(consis_w, torch.Tensor):
            rmdm_loss_it = (
                consis_w.detach() * consistency_loss + diverse_loss * diverse_w.detach()
            )
            if std >= 1e-1:
                rmdm_loss_it = consis_w.detach() * consistency_loss

            if self.target_consistency_metric is not None:
                consis_w_loss = consis_w * (
                    (self.target_consistency_metric - consistency_loss.detach())
                    .detach()
                    .mean()
                )
            if self.target_diverse_metric is not None:
                divers_w_loss = diverse_w * (
                    (self.target_diverse_metric - diverse_loss.detach()).detach().mean()
                )
                pass
        else:
            rmdm_loss_it = consis_w * consistency_loss + diverse_loss * diverse_w
            if std >= 1e-1:
                rmdm_loss_it = consis_w * consistency_loss
        return rmdm_loss_it, consis_w_loss, divers_w_loss

    def rmdm_loss_timing(
        self,
        predicted_env_vector: torch.Tensor,
        tasks: torch.Tensor,
        valid: torch.Tensor,
        consis_w: torch.Tensor,
        diverse_w: torch.Tensor,
        need_all_repre: bool = False,
        need_parameter_loss: bool = False,
        rbf_radius: float = 3000.0,
        kernel_type: str = "rbf",
    ):
        if self.current_env_mean is None:
            self.current_env_mean = torch.zeros(
                (self.max_env_len, 1, predicted_env_vector.shape[-1]),
                device=predicted_env_vector.device,
            )
            self.history_env_mean = torch.zeros(
                (self.max_env_len, 1, predicted_env_vector.shape[-1]),
                device=predicted_env_vector.device,
            )
        tasks = tasks[..., -1, 0]  
        tasks_sorted, indices = torch.sort(tasks)
        tasks_sorted_np = tasks_sorted.detach().cpu().numpy().reshape((-1))
        task_ind_map = {}
        tasks_sorted_np_idx = np.where(np.diff(tasks_sorted_np))[0] + 1
        last_ind = 0
        for i, item in enumerate(tasks_sorted_np_idx):
            task_ind_map[tasks_sorted_np[item - 1]] = [last_ind, item]
            last_ind = item
            if i == len(tasks_sorted_np_idx) - 1:
                task_ind_map[tasks_sorted_np[-1]] = [last_ind, len(tasks_sorted_np)]
        predicted_env_vector = predicted_env_vector[indices]
        # remove the invalid data
        if 0 in task_ind_map:
            predicted_env_vector = predicted_env_vector[task_ind_map[0][1] :]
            start_ind = task_ind_map[0][1]
            task_ind_map.pop(0)
            for k in task_ind_map:
                task_ind_map[k][0] -= start_ind
                task_ind_map[k][1] -= start_ind
        if len(task_ind_map) <= 1:
            print(f"current task num: {len(task_ind_map)}, {task_ind_map}")
            return None, None, None, 0
        total_trasition_num = predicted_env_vector.shape[0]
        all_valids, mean_vector, valid_num_list, all_predicted_env_vectors = (
            [],
            [],
            [],
            [],
        )
        real_all_tasks = sorted(list(task_ind_map.keys()))
        all_tasks, self.lst_tasks = real_all_tasks, real_all_tasks
        use_history_mean = True
        for ind, item in enumerate(all_tasks):
            env_vector_it = predicted_env_vector[
                task_ind_map[item][0] : task_ind_map[item][1]
            ]
            if need_all_repre:
                all_predicted_env_vectors.append(env_vector_it)
            point_num = env_vector_it.shape[0]
            repre_it = env_vector_it.mean(dim=0, keepdim=True)
            if item not in self.mean_vector:
                with torch.no_grad():
                    self.history_env_mean[int(item - 1)] = repre_it
            self.current_env_mean[int(item - 1)] = repre_it
            mean_vector.append(repre_it)
            valid_num_list.append(point_num)
        valid_num_tensor = (
            torch.from_numpy(np.array(valid_num_list))
            .to(device=valid.device, dtype=torch.get_default_dtype())
            .reshape((-1, 1, 1))
        )
        task_set = set(all_tasks)
        with torch.no_grad():
            for k in self.mean_vector:
                if k not in task_set:
                    self.current_env_mean[int(k - 1)] = self.history_env_mean[
                        int(k - 1)
                    ]
        self.current_env_mean = self.current_env_mean.detach()
 
        self.history_env_mean = (
            self.history_env_mean * self.tau + (1 - self.tau) * self.current_env_mean
        ) 
        
        self.t += 1

        for item in all_tasks:
            if item not in self.mean_vector:
                self.mean_vector[item] = 1
        repres = [item[0] for item in mean_vector]
        task_indexs = [index for index in task_set]
        valid_repres_len = len(repres)
        for item in self.mean_vector:
            if item not in task_set:
                task_indexs.append(item)
                repres.append(self.history_env_mean[int(item - 1)])
        
        repre_tensor = torch.cat(repres, 0)
        task_cnt = repre_tensor.shape[0]

        if self.anomaly_detection:
            self.abnormal_batch = repre_tensor.detach().cpu().numpy()
            
        version = 'gauss_infonce'
        if version == 'gauss_infonce':
            kernel_radius = self.aux_kernel_radius
            queries = repre_tensor
            keys = [self.history_env_mean[int(index) - 1] for index in task_indexs]
            keys = torch.cat(keys, dim = 0).detach()
            queries_repeat = queries.unsqueeze(0).repeat(task_cnt, 1, 1)
            keys_repeat = keys.unsqueeze(1).repeat(1, task_cnt, 1).detach()
            distance_matrix = (torch.sum(torch.pow(queries_repeat - keys_repeat, 2), dim = -1) + 1e-6).sqrt()
            kernel_distance_matrix = torch.exp(- distance_matrix / kernel_radius)
            l_pos = torch.diag(distance_matrix).view(-1, 1)
            l_neg = distance_matrix
            logits = torch.cat([l_pos, l_neg], dim = 1)
            labels = torch.zeros(l_pos.shape[0], dtype = torch.long)
            labels = labels.to(distance_matrix.device)
            loss_fn = nn.CrossEntropyLoss()
            aux_loss = loss_fn(- logits / kernel_radius, labels)
        else:
            raise NotImplementedError
        
        dpp_loss = torch.tensor([0.0]).to(device = 'cuda')

        if not use_history_mean:
            with torch.no_grad():
                total_mean = (repre_tensor[:valid_repres_len] * valid_num_tensor).sum(
                    dim=0, keepdim=True
                ) / total_trasition_num
            total_outter_var = (
                (repre_tensor[:valid_repres_len] - total_mean).pow(2) * valid_num_tensor
            ).sum(dim=0, keepdim=True) / total_trasition_num
            total_var = (
                (predicted_env_vector - total_mean).pow(2).mean(dim=0, keepdim=True)
            )
            var = max(total_var.mean() - total_outter_var.mean(), 0)
        else:
            total_var = 0
            for ind, item in enumerate(all_tasks):
                mean_vector = self.history_env_mean[int(item - 1)]
                if need_all_repre:
                    env_vector_it = all_predicted_env_vectors[ind]
                else:
                    env_vector_it = predicted_env_vector[
                        task_ind_map[item][0] : task_ind_map[item][1]
                    ]
                var_it = (
                    (env_vector_it - mean_vector.detach())
                    .pow(2)
                    .sum(dim=0, keepdim=True)
                    .mean()
                )
                total_var = total_var + var_it
            var = total_var / total_trasition_num

        stds = var.sqrt()
        consistency_loss = stds 
        self.tracing_consis_loss.append(consistency_loss.item())
        if stds < 1e-3:
            consistency_loss = consistency_loss.detach()
        rmdm_loss_it, consis_w_loss, diverse_w_loss = self.construct_loss(
            consistency_loss, dpp_loss, consis_w, diverse_w, stds.item()
        )
        # add aux loss
        rmdm_loss_it += aux_loss
        if need_parameter_loss:
            if need_all_repre:
                return (
                    rmdm_loss_it,
                    consistency_loss,
                    dpp_loss,
                    aux_loss,
                    len(all_tasks),
                    consis_w_loss,
                    diverse_w_loss,
                    all_predicted_env_vectors,
                    all_valids,
                )
            return (
                rmdm_loss_it,
                consistency_loss,
                dpp_loss,
                len(all_tasks),
                consis_w_loss,
                diverse_w_loss,
            )
        if need_all_repre:
            return (
                rmdm_loss_it,
                consistency_loss,
                dpp_loss,
                len(all_tasks),
                all_predicted_env_vectors,
                all_valids,
            )
        return rmdm_loss_it, consistency_loss, dpp_loss, len(all_tasks),
