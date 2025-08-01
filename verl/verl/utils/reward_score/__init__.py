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
def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None, use_model_reward=False, reward_manager='naive'):
    metadata = ''
    if data_source == 'openai/gsm8k':
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ['lighteval/MATH', 'DigitalLearningGmbH/MATH-lighteval']:
        from . import math_v1
        res = math_v1.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ['codecontests', 'apps', 'codeforces', 'taco', 'open_source_code']:
        from . import prime_code
        public_tests = extra_info.get('public_tests', '')
        if public_tests:
            res, metadata = prime_code.compute_score(solution_str, public_tests, reward_manager, continuous=True)
        else:
            print(f'{data_source} need public_tests!')

    elif "kk" in data_source:
        from . import kk
        res = kk.compute_score(solution_str, ground_truth)
    elif data_source in ["open_source_math", "ys_train"]: # math rule-based reward
        from . import math_v2
        res = math_v2.compute_score(solution_str, ground_truth, reward_manager)
        if res:
            pass
    elif data_source == 'gpqa':
        from . import gpqa
        res = gpqa.compute_score(solution_str, ground_truth, reward_manager)
    elif "deepscaler" in data_source:
        from . import deepscaler
        res = deepscaler.compute_score(solution_str, ground_truth, extra_info)
    else:
        print(data_source)
        raise NotImplementedError

    if isinstance(res, (int, float, bool)):
        return float(res), metadata
    else:
        return float(res[0]), metadata
