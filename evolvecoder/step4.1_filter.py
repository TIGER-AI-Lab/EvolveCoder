import os
import datasets
import json
import random
from typing import List, Optional, Tuple
from collections import defaultdict
import numpy as np
from fire import Fire
from tqdm import tqdm
from pathlib import Path

from evolvecoder.utils import (
    parse_incomplete_json,
    append_jsonl,
    load_jsonl,
    chunking,
    get_python_code_from_string,
    hash_messages,
    pretty_name
)

FILE_NAME = Path(__file__).stem

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


def group_by_pass_pattern(test_pass_matrix, filtered_indexes):
    groups = defaultdict(list)
    for i, passes in enumerate(test_pass_matrix):
        if i not in filtered_indexes:
            groups[tuple(passes)].append(i)
    return groups


def get_duplicate_indexes(groups):
    duplicate_indexes = set()
    for _, indices in groups.items():
        if len(indices) > 5:
            kept = random.sample(indices, 5)
            duplicate_indexes |= {idx for idx in indices if idx not in kept}
    return duplicate_indexes


def update_item_with_filtered_tests(item, all_removed_indexes):
    filtered_tests = [
        test for i, test in enumerate(item['tests'])
        if i not in all_removed_indexes
    ]

    item['raw_tests'] = item['tests']
    item['filtered_tests'] = filtered_tests
    item.pop('tests', None)
    
    if filtered_tests:
        for eval_result in item['gen_result']['eval_results']:
            filtered_statuses = [
                status for i, status in enumerate(eval_result['test_cases_pass_status'])
                if i not in all_removed_indexes
            ]
            eval_result['test_cases_pass_status'] = filtered_statuses
            passes = [s['pass'] for s in filtered_statuses]
            eval_result['pass_rate'] = sum(passes) / len(passes) if passes else 0.0


def compute_test_case_diversity(item):
    arr = [
        [x['pass'] for x in eval_result['test_cases_pass_status']]
        for eval_result in item['gen_result']['eval_results']
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
    groups = group_by_pass_pattern(matrix, filtered_indexes)
    duplicates = get_duplicate_indexes(groups)
    all_removed = set(filtered_indexes) | duplicates

    update_item_with_filtered_tests(item, all_removed)

    item['gen_result']['test_case_diversity'] = compute_test_case_diversity(item)

    return item


FILE_NAME = Path(__file__).stem

def main(
    file_path: str,
    num_proc: int = 1,
    output_dir: str = None,
    round: int = 1,
    overwrite: bool = False,
    min_test_cases: int = 5,
    max_perfect_solutions: int = 60
):
    assert os.path.exists(file_path), f"File {file_path} does not exist."
    output_dir = output_dir or Path(file_path).parent
    output_file = output_dir / f"{FILE_NAME}_round_{round}.jsonl"
    if output_file.exists() and output_file.stat().st_size != 0 and not overwrite:
        print(f"Output file {output_file} already exists and is not empty. Use --overwrite to overwrite.")
        return

    dataset = datasets.Dataset.from_json(file_path)
    
    num_tests_before = [len(x['tests']) for x in tqdm(dataset, desc="Calculating avg test cases before filtering")]
    avg_tests_before = np.mean(num_tests_before)
    
    print(f"\n=== Before Filtering ===")
    print(f"Average number of test cases: {avg_tests_before:.2f}")
    print(f"Total items: {len(dataset)}")
    
    dataset = dataset.map(
        filter_test_cases,
        num_proc=num_proc,
        desc="Filtering tests",
        writer_batch_size=100,  # 添加这个参数
        keep_in_memory=False,   # 添加这个参数
    )
    
    num_before_min_filter = len(dataset)
    dataset = dataset.filter(
        lambda x: len(x['filtered_tests']) >= min_test_cases,
        num_proc=num_proc,
        desc=f"Removing items with fewer than {min_test_cases} test cases"
    )
    num_after_min_filter = len(dataset)
    print(f"Removed {num_before_min_filter - num_after_min_filter} items with fewer than {min_test_cases} test cases")
    
    def count_perfect_solutions(item):
        perfect_count = sum(
            1 for eval_result in item['gen_result']['eval_results']
            if eval_result.get('pass_rate', 0) == 1.0
        )
        return perfect_count <= max_perfect_solutions
    
    num_before_perfect_filter = len(dataset)
    dataset = dataset.filter(
        count_perfect_solutions,
        num_proc=num_proc,
        desc=f"Removing items with more than {max_perfect_solutions} perfect solutions"
    )
    num_after_perfect_filter = len(dataset)
    print(f"Removed {num_before_perfect_filter - num_after_perfect_filter} items with more than {max_perfect_solutions} perfect solutions")
    
    num_tests_after = [len(x['filtered_tests']) for x in tqdm(dataset, desc="Calculating avg test cases after filtering")]
    avg_tests_after = np.mean(num_tests_after)
    print(f"\n=== After Filtering ===")
    print(f"Total items: {len(dataset)}")
    print(f"Average number of test cases: {avg_tests_after:.2f}")
    print(f"\n=== Difference ===")
    print(f"Test cases filtered: {avg_tests_before - avg_tests_after:.2f} ({(avg_tests_before - avg_tests_after) / avg_tests_before * 100:.1f}%)")

    dataset.to_json(output_file, orient="records", lines=True)
    print(f"Processed dataset saved to {output_file}")

if __name__ == "__main__":
    Fire(main)


"""
python evolvecoder/step4.1_filter.py evolvecoder/outputs/all_20_round1/gpt_4.1_mini/step3.8_eval_round_1.jsonl --num_proc 16 --round 1
"""