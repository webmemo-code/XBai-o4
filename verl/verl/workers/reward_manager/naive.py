# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
from verl.utils.reward_score.length_penalty import cal_length_penalty
import torch
import os
import json

class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, config, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        self.config = config

    def __call__(self, data: DataProto, global_steps, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        response_length = []
        total_score = []
        outlines = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            response_length.append(valid_response_length.tolist())
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            # prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            # response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            question = data_item.non_tensor_batch['question']
            score, metadata = self.compute_score(
                data_source=data_source,
                solution_str=sequences_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                use_model_reward=self.config.trainer.use_model_reward,
                reward_manager='naive'
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            # reward_tensor[i, valid_response_length - 1] = reward
            total_score.append(score)
            outlines.append({'question': question, 'solution': ground_truth,'pred': sequences_str, 'score': score, 'metadata': metadata, 'global_steps': global_steps})
            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", sequences_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if self.config.trainer.length_penalty:
            reward_tensor, length_reward = cal_length_penalty(response_length, total_score, reward_tensor, self.config)
        with open(os.path.join(self.config.trainer.default_local_dir, 'reward.jsonl'), 'a+', encoding='utf-8') as f: # save score
            for i in range(len(data)):
                data_i = outlines[i]
                if self.config.trainer.length_penalty:
                    data_i['length_penalty'] = length_reward[i]
                f.write(json.dumps(data_i, ensure_ascii=False)+'\n')
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor