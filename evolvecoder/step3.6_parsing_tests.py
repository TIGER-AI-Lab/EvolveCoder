import os
import datasets
import json
from typing import List, Optional, Tuple
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
ERROR_QUESTION = "Error in question generation"
ERROR_TESTS = ["assert False"]

def filter_parsed_items(item):
    """
    Filter function to check if the item has 'gpt_response' and 'tests'.
    """
    gpt_response = item['synthesis_result_second'].get('gpt_response', None)
    tests = item['synthesis_result_second'].get('tests', None)
    if gpt_response and gpt_response != ERROR_QUESTION and tests and tests != ERROR_TESTS:
        return True
    return False

def main(
    file_path: str,
    round: int = 0,
    num_proc: int = 1,
    output_dir: str = None,
    do_filter: bool = False,
    overwrite: bool = False
):
    """
    Main function to generate test cases for a given dataset.
    :param dataset_name: Name of the dataset to process.
    :param ct: Number of test cases to generate.
    """
    assert os.path.exists(file_path), f"File {file_path} does not exist."
    output_dir = output_dir or Path(file_path).parent
    output_file = output_dir / f"{FILE_NAME}_round_{round}.jsonl"
    if output_file.exists() and output_file.stat().st_size != 0 and not overwrite:
        print(f"Output file {output_file} already exists and is not empty. Use --overwrite to overwrite.")
        return

    dataset = datasets.Dataset.from_json(file_path)

    def parsing_item(item):
        gpt_response = item['synthesis_result_second'].get('gpt_response', None)
        
        # parse the response
        try:
            obj = parse_incomplete_json(gpt_response)
            tests = obj.get("tests", ERROR_TESTS)
            if not tests:
                tests = ERROR_TESTS
        except Exception as e:
            print(f"Error parsing response: {e}")
            tests = ERROR_TESTS
        
        item['synthesis_result_second']['tests'] = tests
        item.pop('filtered_tests_second', None)
        item.pop('sampled_solutions', None)
        item.pop('eval_matrix', None)
        return item
    
    # Process the dataset in parallel
    dataset = dataset.map(
        parsing_item,
        num_proc=num_proc,
        desc="Parsing dataset",
    )
    if do_filter:
        print(f"Before filtering, dataset size: {len(dataset)}")
        dataset = dataset.filter(
            filter_parsed_items,
            num_proc=num_proc,
            desc="Filtering parsed items",
        )
        print(f"After filtering, dataset size: {len(dataset)}")
    # Save the processed dataset
    dataset.to_json(output_file, orient="records", lines=True)
    print(f"Processed dataset saved to {output_file}")

if __name__ == "__main__":
    Fire(main)

"""
python evolvecoder/step3.7_parsing_tests.py evolvecoder/outputs/all_20_round1/gpt_4.1_mini/step3.6_gen_tests_results_round_1.jsonl --round 1 --num_proc 1
"""