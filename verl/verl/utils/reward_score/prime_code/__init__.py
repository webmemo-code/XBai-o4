# Copyright 2024 PRIME team and/or its affiliates
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

import json
import traceback

from .utils import check_correctness as apps_check_correctness

def extract_solution(solution_str: str, reward_manager: str):
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    # Split response to isolate assistant output
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    elif "<｜Assistant｜>" in solution_str:
        processed_str = solution_str.split("<｜Assistant｜>", 1)[1]
    else:
        if reward_manager == 'prime':
            processed_str = solution_str
        else:
            print("[Error] Failed to locate model response header")
            return None, solution_str

    if '</think>' in processed_str:
        final_answer = processed_str.split('</think>')[-1].lstrip().rstrip('<｜end▁of▁sentence｜>')
    else:
        final_answer = None
    return final_answer, processed_str

def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    # print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        # print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            # print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if positions['think_start'] > positions['think_end']:
        # print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        pass
        # print("  Tag sequence validation passed")
    return validation_passed

def compute_score_per(completion, test_cases, continuous=False):
    # try to get code solution from completion. if the completion is pure code, this will not take effect.
    solution = completion.split("```python")[-1].split("```")[0]
    try:
        try:
            if not isinstance(test_cases, dict):
                test_cases = json.loads(test_cases)
        except Exception as e:
            print(f"Error:{e}")

        # Complete check on all in-out pairs first. If there is no failure, per-sample test can be skipped.
        try:
            res, metadata = apps_check_correctness(in_outs=test_cases, generation=solution, timeout=5, debug=False)
            metadata = dict(enumerate(metadata))[0]
            success = all(map(lambda x: x is True, res))
            if success:
                return success, metadata
        except Exception:
            pass

        test_cases_list = []
        inputs = test_cases["inputs"]
        outputs = test_cases["outputs"]
        for i in range(len(inputs)):
            test_cases_list.append({"inputs": [inputs[i]], "outputs": [outputs[i]]})
        metadata_list = None
        success = False
        if continuous:
            # per sample test: if continuous score is needed, test first 10 samples regardless of failures
            # do not test all samples cuz some problems have enormous test cases
            metadata_list = []
            res_list = []
            for test_case_id, test_case in enumerate(test_cases_list):
                res, metadata = apps_check_correctness(in_outs=test_case, generation=solution, timeout=10, debug=False)
                try:
                    metadata = dict(enumerate(metadata))[0]  # metadata can be empty occasionally
                except Exception:
                    metadata = {}
                metadata["test_case"] = {}
                metadata["test_case"]["input"] = str(test_case["inputs"][0])
                metadata["test_case"]["output"] = str(test_case["outputs"][0])
                metadata["test_case"]["res"] = str(res)
                metadata_list.append(metadata)
                res_list.extend(res)

                if test_case_id >= 9:
                    break
            res_count = len(res_list) if len(res_list) > 0 else 1
            success = sum(map(lambda x: x is True, res_list)) / res_count
    except Exception:
        traceback.print_exc(10)
        success = False
        metadata_list = None
    return success, metadata_list

def compute_score(prediction, test_cases, reward_manager, continuous=False):
    prediction_extract, processed_str  = extract_solution(prediction, reward_manager)
    if reward_manager == 'naive':
        format_correct = validate_response_structure(processed_str)
        format_score = 1 if format_correct else 0
    else:
        format_score = 1
    score = 0
    output = ''
    if format_score and prediction_extract:
        res, metadata_list = compute_score_per(prediction_extract, test_cases, continuous)
        if res == 1:
            score = 1
        else:
            score = 0
        return score, json.dumps(metadata_list, ensure_ascii=False)
    else:
        score = 0.1
        return score, output
