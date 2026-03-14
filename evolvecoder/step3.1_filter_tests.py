import os
import datasets
import random
import json
from typing import List, Optional, Tuple
from collections import defaultdict
import numpy as np
from fire import Fire
from tqdm import tqdm
from pathlib import Path

def build_test_pass_matrix(item):
    num_tests = len(item['tests'])
    matrix = []
    for test_idx in range(num_tests):
        pass_statuses = [
            eval_result['test_cases_pass_status'][test_idx]['pass']
            for eval_result in item['gen_result']['eval_results']
        ]
        matrix.append(pass_statuses)
    return matrix


def get_filtered_test_indexes(test_pass_matrix):
    filtered_indexes = [
        i for i, passes in enumerate(test_pass_matrix)
        if sum(passes) / len(passes) < 0.1
    ]
    return filtered_indexes


def update_item_with_filtered_tests(item, all_removed_indexes):
    filtered_tests = [
        test for i, test in enumerate(item['tests'])
        if i not in all_removed_indexes
    ]

    item['raw_tests'] = item['tests']
    item['filtered_tests_first'] = filtered_tests if filtered_tests else ['']
    item.pop('tests', None)

    if item['filtered_tests_first'] != ['']:
        eval_results_first = []
        for eval_result in item['gen_result']['eval_results']:
            filtered_statuses = [
                status for i, status in enumerate(eval_result['test_cases_pass_status'])
                if i not in all_removed_indexes
            ]
            passes = [s['pass'] for s in filtered_statuses]
            
            new_eval_result = eval_result.copy()
            new_eval_result['test_cases_pass_status'] = filtered_statuses
            new_eval_result['pass_rate'] = sum(passes) / len(passes) if passes else 0.0
            
            eval_results_first.append(new_eval_result)
        
        item['gen_result']['eval_results_first'] = eval_results_first


def compute_test_case_diversity(item):
    arr = [
        [x['pass'] for x in eval_result['test_cases_pass_status']]
        for eval_result in item['gen_result']['eval_results_first']
    ]

    if not arr:
        return {"arr": [], "mean": []}

    try:
        arr = np.array(arr).T.tolist()
        mean_arr = np.mean(arr, axis=1).tolist()
    except Exception as e:
        print(f"[Error in compute_test_case_diversity] {type(e).__name__}: {e}")
        raise

    return {"arr": arr, "mean": mean_arr}


def filter_test_cases(item):
    assert len(item['tests']) == len(item['gen_result']['test_case_diversity']['mean'])

    matrix = build_test_pass_matrix(item)
    filtered_indexes = get_filtered_test_indexes(matrix)
    all_removed = set(filtered_indexes)

    update_item_with_filtered_tests(item, all_removed)

    if item['filtered_tests_first'] != ['']:
        item['gen_result']['test_case_diversity_first'] = compute_test_case_diversity(item)

    return item

FILE_NAME = Path(__file__).stem

def main(
    file_path: str,
    num_proc: int = 1,
    output_dir: str = None,
    round: int = 1,
    overwrite: bool = False
):
    assert os.path.exists(file_path), f"File {file_path} does not exist."
    output_dir = output_dir or Path(file_path).parent
    if round <= 1:
        output_file = output_dir / f"round_{round}" / f"{FILE_NAME}_round_{round}.jsonl"
    else:
        assert Path(output_dir).name == f"round_{round-1}"
        output_dir = Path(output_dir).parent
        output_file = output_dir / f"round_{round}" / f"{FILE_NAME}_round_{round}.jsonl"
    if output_file.exists() and output_file.stat().st_size != 0 and not overwrite:
        print(f"Output file {output_file} already exists and is not empty. Use --overwrite to overwrite.")
        return

    dataset = datasets.Dataset.from_json(file_path)
    
    num_tests_before = [len(x['tests']) for x in tqdm(dataset, desc="Calculating avg test cases before filtering")]
    avg_before = np.mean(num_tests_before)
    print(f"Average number of test cases before filtering: {avg_before:.2f}")
    
    dataset = dataset.map(
        filter_test_cases,
        num_proc=num_proc,
        desc="Filtering test cases",
    )
    
    num_before_removal = len(dataset)
    dataset = dataset.filter(
        lambda x: not (len(x['filtered_tests_first']) == 1 and x['filtered_tests_first'][0] == ''),
        num_proc=num_proc,
        desc="Removing items with empty filtered_tests"
    )
    num_after_removal = len(dataset)
    print(f"Removed {num_before_removal - num_after_removal} items with no valid test cases")

    
    num_tests_after = [len(x['filtered_tests_first']) for x in tqdm(dataset, desc="Calculating avg test cases after filtering")]
    avg_after = np.mean(num_tests_after)
    print(f"Average number of test cases after filtering: {avg_after:.2f}")

    dataset.to_json(output_file, orient="records", lines=True)
    print(f"Processed dataset saved to {output_file}")

if __name__ == "__main__":
    Fire(main)


"""
python acecoderv3_fine_grained_test_cases/step3.1_filter_tests.py acecoderv3_fine_grained_test_cases/outputs/all_20_round1/gpt_4.1_mini/step2.2_eval_Qwen3_32B_seed42_0_81.jsonl --num_proc 16 --round 1
"""