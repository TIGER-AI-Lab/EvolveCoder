import os
import datasets
import json
from typing import List, Optional, Tuple
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

    def filter_test_cases(item):
        assert len(item['synthesis_result']['tests']) == len(item['gen_result']['test_case_diversity']['mean'])
        zero_indexes = [i for i, x in enumerate(item['gen_result']['test_case_diversity']['mean']) if x == 0]
        filtered_tests = [
            test for i, test in enumerate(item['synthesis_result']['tests'])
            if i not in zero_indexes
        ]
        
        if 'filtered_tests' not in item:
            item['filtered_tests'] = []

        item['filtered_tests'].extend(filtered_tests)

        return item
    
    num_tests_before = [len(x['synthesis_result']['tests']) for x in tqdm(dataset, desc="Calculating avg test cases before filtering")]
    avg_before = np.mean(num_tests_before)
    print(f"Average number of test cases before filtering: {avg_before:.2f}")
    
    # Process the dataset in parallel
    dataset = dataset.map(
        filter_test_cases,
        num_proc=num_proc,
        desc="Filtering test cases",
    )
    
    num_tests_after = [len(x.get('filtered_tests', [])) for x in tqdm(dataset, desc="Calculating avg test cases after filtering")]
    avg_after = np.mean(num_tests_after)
    print(f"Average number of test cases after filtering: {avg_after:.2f}")

    dataset.to_json(output_file, orient="records", lines=True)
    print(f"Processed dataset saved to {output_file}")

if __name__ == "__main__":
    Fire(main)


"""
python acecoderv3/step3.1_filter_tests.py acecoderv3/outputs/APPS/gpt_4.1_mini/step2.2_eval_Qwen3_4B_seed42_0_2.jsonl --num_proc 16 --round 1
"""