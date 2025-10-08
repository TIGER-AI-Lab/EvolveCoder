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

    def filter_solutions(item):
        item['filtered_outputs'] = []
        test_case_diversity_arr = []
        for eval_result in item['gen_result']['eval_results']:
            if eval_result['pass_rate'] >= 0.1:
                item['filtered_outputs'].append(eval_result['parse_code'])
            else:
                continue
            passes = [x['pass'] for x in eval_result['test_cases_pass_status']]
            test_case_diversity_arr.append(passes)

        if test_case_diversity_arr:
            test_case_diversity_arr = np.array(test_case_diversity_arr).T.tolist()
            mean_arr = np.mean(test_case_diversity_arr, axis=1).tolist()
        else:
            test_case_diversity_arr, mean_arr = [], []

        item['test_case_diversity'] = {
            "arr": test_case_diversity_arr,
            "mean": mean_arr,
        }

        return item
    
    num_outputs_before = [len(x['gen_result']['eval_results']) for x in tqdm(dataset, desc="Calculating avg outputs before filtering")]
    avg_before = np.mean(num_outputs_before)
    print(f"Average number of outputs before filtering: {avg_before:.2f}")
    
    # Process the dataset in parallel
    dataset = dataset.map(
        filter_solutions,
        num_proc=num_proc,
        desc="Filtering outputs",
    )
    
    num_outputs_after = [len(x['filtered_outputs']) for x in tqdm(dataset, desc="Calculating avg outputs after filtering")]
    avg_after = np.mean(num_outputs_after)
    print(f"Average number of outputs after filtering: {avg_after:.2f}")

    dataset.to_json(output_file, orient="records", lines=True)
    print(f"Processed dataset saved to {output_file}")

if __name__ == "__main__":
    Fire(main)


"""
python acecoderv3/step3.5_filter_solutions.py acecoderv3/outputs/all_20/gpt_4.1_mini/step3.4_combine_eval_tests_round_1.jsonl --num_proc 16 --round 1
"""