import fire
import json
import datasets
import random
from acecoderv2.code_eval import eval_codes, parse_code
from acecoderv2.synthesizer.utils import print_statistics
from typing import List, Union, Optional
from pathlib import Path
import numpy as np
from collections import Counter

FILE_NAME = Path(__file__).stem
LAST_STEP_NAME = "step2.1_gen"

def main(
    file_path: str,
    output_dir: str = None,
    overwrite: bool = False,
    num_proc: int = 64,
    max_samples: Optional[int] = 0
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
    
    if max_samples > 0 and len(data) > max_samples:
        random.seed(42)  # For reproducibility
        random_idxs = set(random.sample(range(len(data)), max_samples))
        data = [data[i] for i in range(len(data)) if i in random_idxs]

    print(f"📥 Loaded {len(data)} problems")

    # Extract solutions and test cases
    solution_strs = []
    test_cases = []
    for item in data:
        for llm_output in item['gen_result']['outputs']:
            solution_strs.append(llm_output)
            test_cases.append(item['synthesis_result']['tests'])

    print(f"🔧 Processing {len(solution_strs)} solutions...")

    dataset = datasets.Dataset.from_dict({
        'solution_str': solution_strs,
        'test_case': test_cases
    })
    
    def parse_code_func(item):
        thinking_end_idx = item['solution_str'].find("</think>")
        if thinking_end_idx != -1:
            item['parse_code'] = item['solution_str'][thinking_end_idx + len("</think>"):]
        else:
            item['parse_code'] = item['solution_str']
        return item
    
    dataset = dataset.map(parse_code_func, num_proc=num_proc, desc="Parsing code")

    print("⚡ Evaluating codes...")
    pass_rates, test_cases_pass_status = eval_codes(
        solution_strs=dataset['parse_code'],
        test_cases=dataset['test_case'],
        return_test_cases_pass_status=True,
        binary=False,
        num_processes=num_proc,
    )

    # Reassign results back to original data structure
    idx = 0
    for item in data:
        item['gen_result']['eval_results'] = []
        test_case_diversity_arr = []
        for _ in range(len(item['gen_result']['outputs'])):
            item['gen_result']['eval_results'].append({
                'pass_rate': pass_rates[idx],
                'test_cases_pass_status': test_cases_pass_status[idx],
                'parse_code': dataset['parse_code'][idx]
            })
            item['gen_result'].pop('outputs', None)  # Remove outputs to save space
            test_case_diversity_arr.append([x['pass'] for x in test_cases_pass_status[idx]])
            idx += 1
        test_case_diversity_arr = np.array(test_case_diversity_arr).T.tolist()
        item['gen_result']['test_case_diversity'] = {
            "arr": test_case_diversity_arr,
            "mean": np.mean(test_case_diversity_arr, axis=1).tolist(),
        }

    

    # Save output data
    print(f"💾 Saving results to: {output_file}")
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Print comprehensive statistics
    print_statistics(data, output_file=stats_output_file)
    print(f"✅ Results saved to {output_file}")
    
    
    
if __name__ == "__main__":
    fire.Fire(main)

"""
python step2.2_eval.py outputs/Magicoder_Evol_Instruct_110K/gpt_4o_mini/step2.1_gen_Qwen2_vllm_seed42_0_50.jsonl
python step2.2_eval.py outputs/Magicoder_Evol_Instruct_110K/gpt_4o_mini/step2.1_gen_gpt_4.1_mini_vllm_seed42_0_50.jsonl
python step2.2_eval.py outputs/Magicoder_Evol_Instruct_110K/o4_mini/step2.1_gen_Qwen3_8B_seed42.jsonl --overwrite True
"""

