import fire
import json
import datasets
from acecoderv2.code_eval import eval_codes, parse_code
from typing import List, Union, Optional
from pathlib import Path
import numpy as np
from collections import Counter
from acecoderv2.synthesizer.utils import print_statistics

FILE_NAME = Path(__file__).stem
LAST_STEP_NAME = "step2.2_eval"


def filter_test_cases(item):
    gen_result = item['gen_result']
    test_case_diversity = gen_result['test_case_diversity']
    test_case_diversity_arr = test_case_diversity['arr']
    to_filter_test_cases_idxs = []
    for i, arr in enumerate(test_case_diversity_arr):
        # filter out test cases that all failed
        if not any(arr):
            to_filter_test_cases_idxs.append(i)
            continue
    
    if to_filter_test_cases_idxs:
        print(f"Filtering out {len(to_filter_test_cases_idxs)} test cases from {item['gen_result']['qid']}")
        for eval_result in gen_result['eval_results']:
            eval_result['test_cases_pass_status'] = [
                status for j, status in enumerate(eval_result['test_cases_pass_status']) 
                if j not in to_filter_test_cases_idxs
            ]
            eval_result['pass_rate'] = np.mean([
                [x['pass'] for x in eval_result['test_cases_pass_status']]
            ]).item() if eval_result['test_cases_pass_status'] else 0.0
        item['synthesis_result']['tests'] = [
            test for j, test in enumerate(item['synthesis_result']['tests'])
            if j not in to_filter_test_cases_idxs
        ]
        test_case_diversity_arr = [test_case_diversity_arr[i] for i in range(len(test_case_diversity_arr)) if i not in to_filter_test_cases_idxs]
        test_case_diversity['arr'] = test_case_diversity_arr
        if len(test_case_diversity_arr) == 0:
            test_case_diversity['mean'] = None
        else:
            test_case_diversity['mean'] = np.mean(test_case_diversity_arr, axis=1).tolist()
    
    return item

def main(
    file_path: str,
    output_dir: str = None,
    overwrite: bool = False,
    num_proc: int = 32,
):
    output_dir = Path(output_dir) if output_dir else Path(file_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    new_file_name = Path(file_path).stem.replace(LAST_STEP_NAME, FILE_NAME)
    output_file = output_dir / f"{new_file_name}.jsonl"
    stats_output_file = output_dir / f"{new_file_name}_stats.txt"
    
    if output_file.exists() and output_file.stat().st_size != 0 and not overwrite:
        print(f"Output file {output_file} already exists. Use --overwrite to overwrite.")
        with open(output_file, 'r') as f:
            data = [json.loads(line) for line in f.readlines()]
        print_statistics(data, output_file=stats_output_file)
        print(f"Returning cached data from {output_file}")
        return

    print(f"🔄 Loading data from: {file_path}")
    
    # Load input data
    if file_path.endswith('.jsonl'):
        with open(file_path, 'r') as f:
            data = [json.loads(line) for line in f.readlines()]
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
    else:
        raise ValueError("Unsupported file format. Please provide a .jsonl or .json file.")

    print(f"📥 Loaded {len(data)} problems")

    # filter test cases
    dataset = datasets.Dataset.from_list(data)
    dataset = dataset.map(filter_test_cases, num_proc=num_proc, desc="Filtering test cases")
    dataset = dataset.filter(lambda item: len(item['synthesis_result']['tests']) > 0, num_proc=num_proc, desc="Removing items with no tests")
    print(f"Before filtering, {len(data)} items, after filtering, {len(dataset)} items")

    # Save output data
    print(f"💾 Saving results to: {output_file}")
    with open(output_file, 'w') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    # Print comprehensive statistics
    print_statistics(dataset, output_file=stats_output_file)
    print(f"✅ Results saved to {output_file}")
    
if __name__ == "__main__":
    fire.Fire(main)

"""
python step2.2_eval.py outputs/Magicoder_Evol_Instruct_110K/gpt_4o_mini/step2.1_gen_Qwen2_vllm_seed42_0_50.jsonl
python step2.2_eval.py outputs/Magicoder_Evol_Instruct_110K/gpt_4o_mini/step2.1_gen_gpt_4.1_mini_vllm_seed42_0_50.jsonl
python step2.2_eval.py outputs/Magicoder_Evol_Instruct_110K/o4_mini/step2.1_gen_Qwen3_8B_seed42.jsonl --overwrite True
"""

