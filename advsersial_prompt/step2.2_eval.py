import fire
import json
import random
import sys
from pathlib import Path
import re
import numpy as np
from collections import Counter

sys.path.append(str(Path(__file__).resolve().parent.parent))

from code_eval import eval_codes
from synthesizer.utils import print_statistics
from typing import List, Optional

FILE_NAME = Path(__file__).stem
LAST_STEP_NAME = "step2.1_gen"  # Also handle merged files
MERGED_STEP_NAME = "step2.1_merged"


def main(
    file_path: str,
    output_dir: str = None,
    overwrite: bool = False,
    num_proc: int = 64,
    max_samples: Optional[int] = None,
    current_round: int = 0
):
    
    output_dir = Path(output_dir) if output_dir else Path(file_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    # Handle both regular generation files and merged files
    file_stem = Path(file_path).stem
    if MERGED_STEP_NAME in file_stem:
        new_file_name = file_stem.replace(MERGED_STEP_NAME, FILE_NAME)
    else:
        new_file_name = file_stem.replace(LAST_STEP_NAME, FILE_NAME)
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
    
    if max_samples is not None and max_samples > 0 and len(data) > max_samples:
        random.seed(42)  # For reproducibility
        data = random.sample(data, max_samples)

    print(f"📥 Loaded {len(data)} problems")

    print(f'----------Preparing evaluation data----------')
    
    items_to_process = []
    for item in data:
        # Get all programs that need to be evaluated
        programs_to_eval = []
        
        # Always include the original program
        original_program = item.get('program')
        if original_program:
            programs_to_eval.append(original_program)
        
        # Include any newly generated programs from step1.1 parsing
        if 'synthesis_result' in item:
            generated_programs_step1 = item['synthesis_result'].get('generated_programs', [])
            if generated_programs_step1:
                programs_to_eval.extend(generated_programs_step1)
        
        # Include any newly generated programs from step2.1 generation
        generated_programs = item.get('gen_result', {}).get('generated_programs', [])
        if generated_programs:
            for gen_prog in generated_programs:
                # Extract the actual code from the response
                if isinstance(gen_prog, dict) and 'message' in gen_prog and 'content' in gen_prog['message']:
                    program_content = gen_prog['message']['content']
                    # Extract Python code from markdown blocks if present
                    if '```python' in program_content:
                        code_blocks = program_content.split('```python')
                        for block in code_blocks[1:]:  # Skip first split (before first code block)
                            code = block.split('```')[0].strip()
                            if code:
                                programs_to_eval.append(code)
                    elif '```' in program_content:
                        code_blocks = program_content.split('```')
                        for i in range(1, len(code_blocks), 2):  # Take odd-indexed blocks (code)
                            code = code_blocks[i].strip()
                            if code:
                                programs_to_eval.append(code)
                    else:
                        # No code blocks, treat the entire content as code
                        programs_to_eval.append(program_content.strip())
                elif isinstance(gen_prog, str):
                    programs_to_eval.append(gen_prog)
        
        if not programs_to_eval:
            print(f"Warning: Skipping item because no programs found. Item: {item.get('gen_result', {}).get('qid')}")
            continue

        # The tests are the combination of original tests and newly generated ones.
        original_tests = item.get('synthesis_result', {}).get('tests', [])
        new_tests_step2_1 = item.get('gen_result', {}).get('outputs', [])
        new_tests_step1_1 = item.get('synthesis_result', {}).get('generated_test_cases', [])
        
        if not isinstance(original_tests, list): original_tests = []
        if not isinstance(new_tests_step2_1, list): new_tests_step2_1 = []
        if not isinstance(new_tests_step1_1, list): new_tests_step1_1 = []

        # Sometimes the generated tests are strings of lists, e.g. "['assert...']". We need to parse them.
        parsed_new_tests = []
        for t in new_tests_step2_1 + new_tests_step1_1:
            if isinstance(t, str):
                # Handle string-encoded list
                if t.strip().startswith('[') and t.strip().endswith(']'):
                    try:
                        # A bit risky, but common for LLM outputs
                        parsed_list = eval(t)
                        if isinstance(parsed_list, list):
                            parsed_new_tests.extend(parsed_list)
                        else:
                            parsed_new_tests.append(t)
                    except:
                        parsed_new_tests.append(t) # Keep as is if parsing fails
                else:
                    parsed_new_tests.append(t)
            elif isinstance(t, list):
                 parsed_new_tests.extend(t)


        all_tests = original_tests + parsed_new_tests

        if not all_tests:
            print(f"Warning: Skipping item with programs but no tests. Item: {item.get('gen_result', {}).get('qid')}")
            continue
        
        # Store the programs and combined tests for processing
        item['programs_to_eval'] = programs_to_eval
        item['all_tests'] = all_tests
        items_to_process.append(item)

    if not items_to_process:
        print("Error: No valid program/test combinations found to evaluate.")
        with open(stats_output_file, 'w') as f:
            f.write("No valid items to evaluate.\n")
        with open(output_file, 'w') as f:
            f.write("")
        return

    print(f"🔧 Processing {len(items_to_process)} solutions...")

    def advanced_parse_code(solution_str: str, test_cases: List[str]) -> str:
        """
        Injects placeholder implementations for undefined classes and functions found in test cases.
        This version also detects method calls on placeholder classes.
        """
        if not isinstance(solution_str, str):
            solution_str = str(solution_str)

        all_tests_str = "\n".join(str(t) for t in test_cases)
        
        # Find class instantiations like `GridMaster(...)`
        class_instantiations = set(re.findall(r'\b([A-Z]\w*)\s*\(', all_tests_str))
        
        # Find function calls like `my_func(...)`
        function_calls = set(re.findall(r'\b([a-z_]\w*)\s*\(', all_tests_str))

        solution_words = set(re.findall(r'\b\w+\b', solution_str))
        common_keywords = {'assert', 'Solution', 'self', 'range', 'len', 'list', 'dict', 'set', 'tuple', 'print', 'int', 'str', 'float', 'bool', 'True', 'False', 'None', 'sorted', 'min', 'max', 'abs', 'sum'}
        
        placeholders = []
        
        # Handle class placeholders
        for class_name in class_instantiations:
            if class_name not in solution_words and class_name not in common_keywords:
                # Find all method calls on this class, e.g., master.canMove(...)
                methods = set(re.findall(r'\w+\.' + class_name + r'\(\w*\)\.(\w+)\(', all_tests_str))
                # A more general regex for obj.method() where obj might be an instance of our class
                methods.update(set(re.findall(r'\w+\.(\w+)\(', all_tests_str)))


                method_defs = "    def __init__(self, *args, **kwargs): pass\n"
                for method in methods:
                    if method not in solution_words:
                         method_defs += f"    def {method}(self, *args, **kwargs): return None\n"

                placeholders.append(f"class {class_name}:\n{method_defs}")

        # Handle function placeholders
        for func_name in function_calls:
            if func_name not in solution_words and func_name not in common_keywords:
                placeholders.append(f"def {func_name}(*args, **kwargs): return None\n")

        # Remove duplicates
        unique_placeholders = list(dict.fromkeys(placeholders))
        injected_code = "\n".join(unique_placeholders) + "\n\n" + solution_str
        if unique_placeholders:
            print(f"--- Injected Code for QID ---")
            print("\n".join(unique_placeholders))
            print("----------------------------")
        return injected_code

    # Prepare lists for eval_codes - now we need to handle multiple programs per item
    all_parsed_codes = []
    all_test_cases = []
    item_program_mapping = []  # Track which programs belong to which items
    
    for item_idx, item in enumerate(items_to_process):
        programs = item['programs_to_eval']
        tests = item['all_tests']
        
        for program in programs:
            parsed_code = advanced_parse_code(program, tests)
            all_parsed_codes.append(parsed_code)
            all_test_cases.append(tests)
            item_program_mapping.append(item_idx)

    print("⚡ Evaluating codes...")
    pass_rates, test_cases_pass_status = eval_codes(
        solution_strs=all_parsed_codes,
        test_cases=all_test_cases,
        return_test_cases_pass_status=True,
        binary=False,
        num_processes=num_proc,
    )

    print(f'----------Processing evaluation results----------')
    
    # Group results back by item
    eval_idx = 0
    for item_idx, item in enumerate(items_to_process):
        num_programs = len(item['programs_to_eval'])
        
        # Collect all eval results for this item
        eval_results = []
        for prog_idx in range(num_programs):
            eval_result = {
                'pass_rate': pass_rates[eval_idx],
                'test_cases_pass_status': test_cases_pass_status[eval_idx],
                'parse_code': all_parsed_codes[eval_idx]
            }
            eval_results.append(eval_result)
            eval_idx += 1
        
        item['gen_result']['eval_results'] = eval_results

        # Calculate test case diversity matrix: shape (num_test_cases, num_programs)
        num_tests = len(item['all_tests'])
        test_case_diversity_arr = []
        
        for test_idx in range(num_tests):
            test_pass_row = []
            for prog_idx in range(num_programs):
                if test_idx < len(eval_results[prog_idx]['test_cases_pass_status']):
                    test_pass_row.append(eval_results[prog_idx]['test_cases_pass_status'][test_idx]['pass'])
                else:
                    test_pass_row.append(False)
            test_case_diversity_arr.append(test_pass_row)
        
        # Calculate mean pass rate for each test case across all programs
        mean_pass_rates = []
        for test_row in test_case_diversity_arr:
            mean_pass_rates.append(sum(test_row) / len(test_row) if test_row else 0.0)
        
        item['gen_result']['test_case_diversity'] = {
            "arr": test_case_diversity_arr,
            "mean": mean_pass_rates,
        }
        
        item['gen_result'].pop('outputs', None)  # Clean up original outputs to avoid confusion

    print(f" Saving {len(items_to_process)} processed results to: {output_file}")
    with open(output_file, 'w') as f:
        for item in items_to_process:
            # Clean up temporary keys before saving
            item.pop('programs_to_eval', None)
            item.pop('all_tests', None)
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print_statistics(items_to_process, output_file=stats_output_file)
    
    # --- Visualization History Logging ---
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    history_file = vis_dir / "visualization_history.jsonl"
    # Use the passed current_round parameter instead of inferring from path
    
    # To ensure we are logging the state of this round, we append the data.
    with open(history_file, 'a') as f:
        for item in items_to_process:
            # Create a serializable copy for logging
            history_item = json.loads(json.dumps(item))
            history_item['round'] = current_round
            # The 'programs' in the history should be all programs that were evaluated
            history_item['programs'] = history_item.get('programs_to_eval', [item.get('program')])
            # The 'tests' should be all tests used in this eval
            history_item['synthesis_result']['tests'] = history_item.get('all_tests', history_item.get('synthesis_result', {}).get('tests', []))
            f.write(json.dumps(history_item) + '\n')
    
    print(f"📊 Appended {len(items_to_process)} evaluation results for round {current_round} to {history_file}")
    # --- End of Visualization History Logging ---
    
    print(f" Results saved to {output_file}")
    
    
    
def get_round_from_path(file_path: str) -> int:
    """Extracts the round number from a file path using regex."""
    match = re.search(r'_round(\d+)', file_path)
    if match:
        return int(match.group(1))
    # Fallback if no round number in filename, useful for initial runs.
    return 0
    
if __name__ == "__main__":
    fire.Fire(main)

"""
python step2.2_eval.py outputs/Magicoder_Evol_Instruct_110K/gpt_4o_mini/step2.1_gen_Qwen2_vllm_seed42_0_50.jsonl
python step2.2_eval.py outputs/Magicoder_Evol_Instruct_110K/gpt_4o_mini/step2.1_gen_gpt_4.1_mini_vllm_seed42_0_50.jsonl
python step2.2_eval.py outputs/Magicoder_Evol_Instruct_110K/o4_mini/step2.1_gen_Qwen3_8B_seed42.jsonl --overwrite True
"""

