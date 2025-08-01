import json
import os
import re
from os import environ

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
        final_answer = processed_str.split('</think>')[-1].lstrip()
    else:
        final_answer = None
    return final_answer, processed_str

def gpqa_postprocess(text: str) -> str:
    ANSWER_PATTERN = r'(?i)ANSWER\s*:\s*([A-D])'
    match = re.search(ANSWER_PATTERN, text)
    if match:
        return match.group(1)
    return None


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

def compute_score(solution_str, ground_truth, reward_manager):
    prediction_extract, processed_str = extract_solution(solution_str, reward_manager)

    solution_text = ground_truth.get('target', '')
    if reward_manager == 'naive':
        format_correct = validate_response_structure(processed_str)
        format_score = 1 if format_correct else 0
    else:
        format_score = 1

    if format_score and prediction_extract:
        prediction_result = gpqa_postprocess(prediction_extract)
        if prediction_result==solution_text:
            return 1
        else:
            return 0
    else:
        return 0
