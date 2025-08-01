import torch
from typing import Union
import copy
import numpy as np

def cal_length_penalty(response_length: list, scores: list[Union[int, float]], reward_tensor: torch.Tensor, config) -> list:
    """cal length penalty value
    
    Args:
        response_length: valid response_length
        scores: reward scores
        
    Returns:
        scores add length penalty
    """
    rollout_n = config.actor_rollout_ref.rollout.n
    l_coef = config.trainer.length_penalty_coeff
    total_batch_size = len(scores)
    lambda_length = []
    length_reward = []
    for i in range(total_batch_size//rollout_n):
        batch_length = response_length[i*rollout_n: (i+1)*rollout_n]
        batch_length_min = min(batch_length)
        batch_length_max = max(batch_length)
        lambda_length.append([batch_length_min, batch_length_max])

    for j in range(total_batch_size):
        group_id = j//rollout_n
        if lambda_length[group_id][0] == lambda_length[group_id][1]:
            length_reward.append(0)
            pass
        else:
            lambda_length_j = 0.5 - (response_length[j]-lambda_length[group_id][0])/(lambda_length[group_id][1]-lambda_length[group_id][0])
            if scores[j] == 1:
                reward_tensor[j, response_length[j] - 1] += l_coef * lambda_length_j
                length_reward.append(l_coef * lambda_length_j)
            else:
                reward_tensor[j, response_length[j] - 1] += l_coef * min(0, lambda_length_j)
                length_reward.append(l_coef * min(0, lambda_length_j))
    return reward_tensor, length_reward

def cal_length_penalty_v2(response_length: list, scores: list[Union[int, float]], reward_tensor: torch.Tensor, config) -> list:
    """cal length penalty value
    
    Args:
        response_length: valid response_length
        scores: reward scores
        
    Returns:
        scores add length penalty
    """
    rollout_n = config.actor_rollout_ref.rollout.n
    l_coef = config.trainer.length_penalty_coeff
    total_batch_size = len(scores)
    lambda_length = []
    length_reward = []
    for i in range(total_batch_size//rollout_n):
        batch_length = response_length[i*rollout_n: (i+1)*rollout_n]

        # 改动1: 最大最小值只计算有score的rollout
        batch_length_new = []
        score_new = scores[i*rollout_n: (i+1)*rollout_n]
        for s_ind in score_new:
            if score_new[s_ind] == 1:
                batch_length_new.append(batch_length[s_ind])
        if len(batch_length_new) > 0:
            batch_length_min = min(batch_length_new)
            batch_length_max = max(batch_length_new)
            lambda_length.append([batch_length_min, batch_length_max])
        else:
            lambda_length.append([0, 0])
        # 改动1结束

    for j in range(total_batch_size):
        group_id = j//rollout_n
        if lambda_length[group_id][0] == lambda_length[group_id][1]:
            length_reward.append(0)
            pass
        else:
            lambda_length_j = 0.5 - (response_length[j]-lambda_length[group_id][0])/(lambda_length[group_id][1]-lambda_length[group_id][0])
            if scores[j] == 1:
                reward_tensor[j, response_length[j] - 1] += l_coef * lambda_length_j
                length_reward.append(l_coef * lambda_length_j)
            else:
                # 改动2：算错不给惩罚
                reward_tensor[j, response_length[j] - 1] += 0
                length_reward.append(0)
                # 改动2结束
    return reward_tensor, length_reward
#py3langid