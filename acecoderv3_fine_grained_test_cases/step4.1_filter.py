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

from acecoderv2.synthesizer.utils import (
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


def get_filtered_outputs_indexes(item):
    filtered_indexes = []
    for idx, eval_result in enumerate(item['gen_result']['eval_results']):
        pass_rate = eval_result['pass_rate']
        if pass_rate < 0.1:
            filtered_indexes.append(idx)
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
        if len(indices) > 1:
            kept = random.choice(indices)
            duplicate_indexes |= {idx for idx in indices if idx != kept}
    return duplicate_indexes


def update_item_with_filtered_tests(item, all_removed_indexes):
    filtered_tests = [
        test for i, test in enumerate(item['tests'])
        if i not in all_removed_indexes
    ]

    if not filtered_tests:
        filtered_tests = ['assert False']
        for eval_result in item['gen_result']['eval_results']:
            eval_result['test_cases_pass_status'].append({'pass': False})

    item['raw_tests'] = item['tests']
    item['filtered_tests'] = filtered_tests
    item.pop('tests', None)


def update_item_with_filtered_outputs(item, filtered_solution_indexes):
    num_solutions = len(item['outputs'])
    
    kept_indexes = [i for i in range(num_solutions) if i not in filtered_solution_indexes]
    
    item['raw_outputs'] = item['outputs']
    item['filtered_outputs'] = [
        item['outputs'][i] for i in kept_indexes
    ]
    item.pop('outputs', None)
    item.pop('gen_result', None)


def filter_tests_and_solutions(item):
    matrix = build_test_pass_matrix(item)
    filtered_test_indexes = get_filtered_test_indexes(matrix)
    groups = group_by_pass_pattern(matrix, filtered_test_indexes)
    duplicates = get_duplicate_indexes(groups)
    all_removed_tests = set(filtered_test_indexes) | duplicates
    
    update_item_with_filtered_tests(item, all_removed_tests)
    
    filtered_output_indexes = get_filtered_outputs_indexes(item)
    update_item_with_filtered_outputs(item, filtered_output_indexes)
    
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
    output_file = output_dir / f"{FILE_NAME}_round_{round}.jsonl"
    if output_file.exists() and output_file.stat().st_size != 0 and not overwrite:
        print(f"Output file {output_file} already exists and is not empty. Use --overwrite to overwrite.")
        return

    dataset = datasets.Dataset.from_json(file_path)
    
    num_tests_before = [len(x['tests']) for x in tqdm(dataset, desc="Calculating avg test cases before filtering")]
    num_solutions_before = [len(x['outputs']) for x in tqdm(dataset, desc="Calculating avg outputs before filtering")]
    avg_tests_before = np.mean(num_tests_before)
    avg_solutions_before = np.mean(num_solutions_before)
    
    print(f"\n=== Before Filtering ===")
    print(f"Average number of test cases: {avg_tests_before:.2f}")
    print(f"Average number of solutions: {avg_solutions_before:.2f}")
    
    dataset = dataset.map(
        filter_tests_and_solutions,
        num_proc=num_proc,
        desc="Filtering tests and solutions",
    )
    
    num_tests_after = [len(x['filtered_tests']) for x in tqdm(dataset, desc="Calculating avg test cases after filtering")]
    num_solutions_after = [len(x['filtered_outputs']) for x in tqdm(dataset, desc="Calculating avg outputs after filtering")]
    avg_tests_after = np.mean(num_tests_after)
    avg_solutions_after = np.mean(num_solutions_after)
    
    print(f"\n=== After Filtering ===")
    print(f"Average number of test cases: {avg_tests_after:.2f}")
    print(f"Average number of outputs: {avg_solutions_after:.2f}")
    
    print(f"\n=== Difference ===")
    print(f"Test cases filtered: {avg_tests_before - avg_tests_after:.2f} ({(avg_tests_before - avg_tests_after) / avg_tests_before * 100:.1f}%)")
    print(f"Outputs filtered: {avg_solutions_before - avg_solutions_after:.2f} ({(avg_solutions_before - avg_solutions_after) / avg_solutions_before * 100:.1f}%)")


    dataset.to_json(output_file, orient="records", lines=True)
    print(f"Processed dataset saved to {output_file}")

if __name__ == "__main__":
    Fire(main)


"""
python acecoderv3_fine_grained_test_cases/step4.1_filter.py acecoderv3_fine_grained_test_cases/outputs/all_20_round1/gpt_4.1_mini/step3.8_eval_round_1.jsonl --num_proc 16 --round 1
"""