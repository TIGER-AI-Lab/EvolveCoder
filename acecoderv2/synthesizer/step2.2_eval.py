import fire
import json
import datasets
from acecoderv2.code_eval import eval_codes, parse_code
from typing import List, Union, Optional
from pathlib import Path
import numpy as np
from collections import Counter

FILE_NAME = Path(__file__).stem
LAST_STEP_NAME = "step2.1_gen"
def print_statistics(data, pass_rates, test_cases_pass_status):
    """Print comprehensive statistics about the evaluation results"""
    print("\n" + "="*80)
    print("EVALUATION STATISTICS")
    print("="*80)
    
    # Basic statistics
    total_solutions = len(pass_rates)
    total_problems = len(data)
    solutions_per_problem = total_solutions // total_problems if total_problems > 0 else 0
    
    print(f"📊 Basic Info:")
    print(f"   Total problems: {total_problems}")
    print(f"   Total solutions: {total_solutions}")
    print(f"   Solutions per problem: {solutions_per_problem}")
    
    # Pass rate statistics
    pass_rates_array = np.array(pass_rates)
    print(f"\n🎯 Pass Rate Statistics:")
    print(f"   Mean pass rate: {pass_rates_array.mean():.4f} ({pass_rates_array.mean()*100:.2f}%)")
    print(f"   Median pass rate: {np.median(pass_rates_array):.4f} ({np.median(pass_rates_array)*100:.2f}%)")
    print(f"   Std deviation: {pass_rates_array.std():.4f}")
    print(f"   Min pass rate: {pass_rates_array.min():.4f} ({pass_rates_array.min()*100:.2f}%)")
    print(f"   Max pass rate: {pass_rates_array.max():.4f} ({pass_rates_array.max()*100:.2f}%)")
    
    # Percentiles
    percentiles = [25, 50, 75, 90, 95, 99]
    print(f"   Percentiles:")
    for p in percentiles:
        val = np.percentile(pass_rates_array, p)
        print(f"     {p}th: {val:.4f} ({val*100:.2f}%)")
    
    # Pass rate distribution
    perfect_solutions = np.sum(pass_rates_array == 1.0)
    zero_solutions = np.sum(pass_rates_array == 0.0)
    partial_solutions = total_solutions - perfect_solutions - zero_solutions
    
    print(f"\n✅ Solution Quality Distribution:")
    print(f"   Perfect solutions (100% pass): {perfect_solutions} ({perfect_solutions/total_solutions*100:.2f}%)")
    print(f"   Partial solutions (0% < pass < 100%): {partial_solutions} ({partial_solutions/total_solutions*100:.2f}%)")
    print(f"   Failed solutions (0% pass): {zero_solutions} ({zero_solutions/total_solutions*100:.2f}%)")
    
    # Pass rate bins
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(pass_rates_array, bins=bins)
    print(f"\n📈 Pass Rate Distribution (bins):")
    for i in range(len(bins)-1):
        count = hist[i]
        percentage = count/total_solutions*100
        print(f"   [{bins[i]:.1f}, {bins[i+1]:.1f}): {count} solutions ({percentage:.2f}%)")
    
    # Problem-level analysis
    print(f"\n🧩 Problem-level Analysis:")
    idx = 0
    problem_pass_rates = []
    problem_best_pass_rates = []
    problem_worst_pass_rates = []
    
    for i, item in enumerate(data):
        num_solutions = len(item['gen_result']['outputs'])
        problem_rates = pass_rates[idx:idx+num_solutions]
        
        avg_rate = np.mean(problem_rates)
        best_rate = np.max(problem_rates)
        worst_rate = np.min(problem_rates)
        
        problem_pass_rates.append(avg_rate)
        problem_best_pass_rates.append(best_rate)
        problem_worst_pass_rates.append(worst_rate)
        
        idx += num_solutions
    
    problem_pass_rates = np.array(problem_pass_rates)
    problem_best_pass_rates = np.array(problem_best_pass_rates)
    problem_worst_pass_rates = np.array(problem_worst_pass_rates)
    
    print(f"   Average pass rate per problem: {problem_pass_rates.mean():.4f} ± {problem_pass_rates.std():.4f}")
    print(f"   Best solution per problem: {problem_best_pass_rates.mean():.4f} ± {problem_best_pass_rates.std():.4f}")
    print(f"   Worst solution per problem: {problem_worst_pass_rates.mean():.4f} ± {problem_worst_pass_rates.std():.4f}")
    
    # Problem difficulty analysis
    easy_problems = np.sum(problem_best_pass_rates >= 0.8)
    medium_problems = np.sum((problem_best_pass_rates >= 0.4) & (problem_best_pass_rates < 0.8))
    hard_problems = np.sum(problem_best_pass_rates < 0.4)
    
    print(f"\n🎚️  Problem Difficulty (based on best solution):")
    print(f"   Easy (≥80% pass): {easy_problems} ({easy_problems/total_problems*100:.2f}%)")
    print(f"   Medium (40-80% pass): {medium_problems} ({medium_problems/total_problems*100:.2f}%)")
    print(f"   Hard (<40% pass): {hard_problems} ({hard_problems/total_problems*100:.2f}%)")
    
    # Test case analysis
    if test_cases_pass_status:
        print(f"\n🧪 Test Case Analysis:")
        total_test_cases = 0
        passed_test_cases = 0
        failed_test_cases = 0
        error_test_cases = 0
        timeout_test_cases = 0
        
        failure_reasons = Counter()
        error_messages = Counter()
        time_limits = []
        
        for test_status in test_cases_pass_status:
            if test_status:  # if not None/empty
                total_test_cases += len(test_status)
                
                for test_result in test_status:
                    if isinstance(test_result, dict):
                        # Handle dict format with detailed results
                        if test_result.get("pass", False):
                            passed_test_cases += 1
                        else:
                            failed_test_cases += 1
                            
                        # Collect failure reasons
                        reason = test_result.get("reason", "unknown")
                        failure_reasons[reason] += 1
                        
                        # Collect error messages (if present)
                        error_msg = test_result.get("error_message")
                        if error_msg:
                            error_test_cases += 1
                            # Truncate long error messages for counting
                            short_error = str(error_msg)[:100] + "..." if len(str(error_msg)) > 100 else str(error_msg)
                            error_messages[short_error] += 1
                        
                        # Check for timeouts
                        if reason == "timeout" or "timeout" in str(reason).lower():
                            timeout_test_cases += 1
                            
                        # Collect time limits
                        time_limit = test_result.get("time_limit")
                        if time_limit is not None:
                            time_limits.append(time_limit)
                    else:
                        # Handle simple boolean format (backward compatibility)
                        if test_result:
                            passed_test_cases += 1
                        else:
                            failed_test_cases += 1
        
        if total_test_cases > 0:
            overall_test_pass_rate = passed_test_cases / total_test_cases
            print(f"   Total test cases: {total_test_cases}")
            print(f"   Passed test cases: {passed_test_cases} ({passed_test_cases/total_test_cases*100:.2f}%)")
            print(f"   Failed test cases: {failed_test_cases} ({failed_test_cases/total_test_cases*100:.2f}%)")
            print(f"   Overall test pass rate: {overall_test_pass_rate:.4f} ({overall_test_pass_rate*100:.2f}%)")
            
            # Test cases per solution
            test_counts = [len(status) if status else 0 for status in test_cases_pass_status]
            test_counts_array = np.array(test_counts)
            if len(test_counts_array) > 0:
                print(f"   Avg test cases per solution: {test_counts_array.mean():.2f} ± {test_counts_array.std():.2f}")
                print(f"   Min/Max test cases: {test_counts_array.min()}/{test_counts_array.max()}")
            
            # Failure reason analysis
            if failure_reasons:
                print(f"\n   📋 Failure Reasons:")
                for reason, count in failure_reasons.most_common(10):
                    percentage = count/total_test_cases*100
                    print(f"     {reason}: {count} ({percentage:.2f}%)")
            
            # Error analysis
            if error_test_cases > 0:
                print(f"\n   ❌ Error Analysis:")
                print(f"     Test cases with errors: {error_test_cases} ({error_test_cases/total_test_cases*100:.2f}%)")
                if error_messages:
                    print(f"     Top error messages:")
                    for error_msg, count in error_messages.most_common(5):
                        percentage = count/error_test_cases*100
                        print(f"       {count}x ({percentage:.1f}%): {error_msg}")
            
            # Timeout analysis
            if timeout_test_cases > 0:
                print(f"\n   ⏱️  Timeout Analysis:")
                print(f"     Timeout test cases: {timeout_test_cases} ({timeout_test_cases/total_test_cases*100:.2f}%)")
            
            # Time limit analysis
            if time_limits:
                time_limits_array = np.array(time_limits)
                print(f"\n   ⏰ Time Limit Analysis:")
                print(f"     Average time limit: {time_limits_array.mean():.2f}s ± {time_limits_array.std():.2f}s")
                print(f"     Min/Max time limit: {time_limits_array.min():.2f}s / {time_limits_array.max():.2f}s")
                
                # Time limit distribution
                unique_limits, counts = np.unique(time_limits_array, return_counts=True)
                print(f"     Time limit distribution:")
                for limit, count in zip(unique_limits, counts):
                    percentage = count/len(time_limits_array)*100
                    print(f"       {limit}s: {count} tests ({percentage:.2f}%)")
    
    # Top and bottom performing problems
    print(f"\n🏆 Top 5 Easiest Problems (by best solution pass rate):")
    top_indices = np.argsort(problem_best_pass_rates)[-5:][::-1]
    for rank, idx in enumerate(top_indices, 1):
        print(f"   {rank}. Problem {idx}: {problem_best_pass_rates[idx]:.4f} (avg: {problem_pass_rates[idx]:.4f})")
    
    print(f"\n💀 Top 5 Hardest Problems (by best solution pass rate):")
    bottom_indices = np.argsort(problem_best_pass_rates)[:5]
    for rank, idx in enumerate(bottom_indices, 1):
        print(f"   {rank}. Problem {idx}: {problem_best_pass_rates[idx]:.4f} (avg: {problem_pass_rates[idx]:.4f})")
    
    print("="*80)


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
    
    if output_file.exists() and not output_file.stat().st_size and not overwrite:
        print(f"Output file {output_file} already exists. Use --overwrite to overwrite.")
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
        for _ in range(len(item['gen_result']['outputs'])):
            item['gen_result']['eval_results'].append({
                'pass_rate': pass_rates[idx],
                'test_cases_pass_status': test_cases_pass_status[idx],
                'parse_code': dataset['parse_code'][idx]
            })
            idx += 1

    # Print comprehensive statistics
    print_statistics(data, pass_rates, test_cases_pass_status)
        
    # Save output data
    print(f"💾 Saving results to: {output_file}")
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    # with open(output_file.with_suffix('.json'), 'w') as f:
    #     json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"✅ Results saved to {output_file}")

if __name__ == "__main__":
    fire.Fire(main)

"""
python step2.2_eval.py outputs/Magicoder_Evol_Instruct_110K/gpt_4o_mini/step2.1_gen_Qwen2_vllm_seed42_0_50.jsonl
python step2.2_eval.py outputs/Magicoder_Evol_Instruct_110K/gpt_4o_mini/step2.1_gen_gpt_4.1_mini_vllm_seed42_0_50.jsonl
"""

