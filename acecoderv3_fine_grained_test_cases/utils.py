import json
import os
import sys
import time
import hashlib
import regex as re
import numpy as np
from typing import Any, Dict, List, Union
from collections import Counter
from pathlib import Path
def load_jsonl(file_path: str) -> List[Dict[Any, Any]]:
    """load a .jsonl file. Return a List of dictionary, where each dictionary is a line in the file"""
    if not os.path.exists(file_path):
        raise Exception(f"{file_path} Does not exist!!!!")
    with open(file_path, "r") as f:
        lst = f.readlines()
    lst = [json.loads(i) for i in lst]
    return lst


def get_python_code_from_string(input: str) -> str:
    """Basically find code wrapped in ```python ... ``` and return it. If none is found then will return the
    empty string"""
    left_index = input.find("```python")
    if left_index < 0:
        return ""
    input = input[left_index + 9 :]
    right_index = input.find("```")
    if right_index < 0:
        return ""
    input = input[:right_index]
    return input


def parse_incomplete_json(input: str) -> Any:
    """A helper function that will:
    1. Try to parse the whole thing as JSON
    2. Try to find JSON object wrapped in ```json ... ``` or ```python ... ```
    3. Try to parse incomplete JSON (missing ] or }) by progressively trimming

    Only handles missing trailing brackets; modify if you need more cases.
    """
    input = input.strip()

    for marker in ("```json", "```python"):
        left_idx = input.find(marker)
        if left_idx >= 0:
            input = input[left_idx + len(marker):]
            break

    right_idx = input.rfind("```")
    if right_idx >= 0:
        input = input[:right_idx]

    try:
        return json.loads(input)
    except Exception:
        pass

    while len(input) > 0:
        try:
            return json.loads(input + "]}")
        except json.decoder.JSONDecodeError:
            input = input[:-1]

    return {"question": None, "tests": None}



def remove_print_statements_from_python_program(input: str) -> str:
    lst = input.splitlines()
    lst = [i for i in lst if not i.strip().startswith("print")]
    return "\n".join(lst)


def print_data(file: str, idx: int = 0):
    data = load_jsonl(file)
    data = [row for row in data if row["id"] == idx][0]
    for key in data:
        print(f"----------------{key}:-------------------")
        if type(data[key]) == list:
            for i in data[key]:
                if type(i) == list:
                    # we omit the original inferences for easier print statements
                    for ii in i:
                        print(ii)
                    break
                else:
                    print(i)
            print(f"Contained {len(data[key])} items-----")
        else:
            print(data[key])

def chunking(lst: List[Any], n: int) -> List[List[Any]]:
    """Split a list into a list of list where each sublist is of size n"""
    if n <= 0:
        raise Exception(f"Are you fucking kidding me with n = {n}?")
    if len(lst) <= n:
        return [lst]
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def load_jsonl(file_path: str) -> List[Dict[Any, Any]]:
    """load a .jsonl file. Return a List of dictionary, where each dictionary is a line in the file"""
    if not os.path.exists(file_path):
        raise Exception(f"{file_path} Does not exist!!!!")
    with open(file_path, "r") as f:
        lst = f.readlines()
    output = [json.loads(i) for i in lst]
    return output


def load_cache(cache_file: Path):
    if not cache_file.exists():
        return {}
    existing_cache = load_jsonl(cache_file)
    return {item['hash_id']: item for item in existing_cache}


def save_gen_results(final_results, output_file, cache_file):
    with open(output_file, "w") as f:
        for item in final_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    if cache_file.exists():
        os.remove(cache_file)
        
        
def prepare_environment(file_name: str, file_path: str, output_dir: str, round: int, overwrite: bool):
    output_dir = Path(output_dir) if output_dir else Path(file_path).parent
    cache_file = output_dir / f"{file_name}_round_{round}.cache.jsonl"
    output_file = output_dir / f"{file_name}_results_round_{round}.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_file.exists() and not overwrite:
        raise FileExistsError(f"Output file {output_file} exists. Use --overwrite to overwrite.")

    return cache_file, output_file


def save_jsonl(file_path: str, content: List[Dict[Any, Any]]) -> None:
    """save a .jsonl file."""
    with open(file_path, "w") as f:
        for i in content:
            f.write(json.dumps(i) + "\n")


def append_jsonl(file_path: str, content: List[Dict[Any, Any]]) -> None:
    """append to a .jsonl file."""
    with open(file_path, "a") as f:
        for i in content:
            f.write(json.dumps(i) + "\n")


class MyTimer:
    """A simple timer class where you initialize it, then just call print_runtime everytime you want to time yourself"""

    def __init__(self) -> None:
        self.start = time.time()

    def print_runtime(self, message: str, reset_timer: bool = True) -> None:
        """Print the runtime, the output will be in the form of f"{message} took ..."

        Parameter:
            message: a string indicating what you have done
            reset_timer: whether to reset timer so that next call to this function will show the time in between print_runtime
        """
        runtime = time.time() - self.start
        minute = int(runtime / 60)
        seconds = runtime % 60
        if minute > 0:
            print(f"{message} took {minute} minutes {seconds} seconds")
        else:
            print(f"{message} took {seconds} seconds")

        if reset_timer:
            self.start = time.time()




def hash_messages(messages: Union[str, List[Dict[str, Any]]]) -> str:
    """
    Hash the messages to get a unique identifier for the conversation.
    If messages is a string, it will be hashed directly.
    If messages is a list of dictionaries in openai format, it will be converted to a string and then hashed.
    
    Args:
        messages: Either a string or a list of message dictionaries in OpenAI format
                 (e.g., [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}])
    
    Returns:
        str: SHA-256 hash of the messages as a hexadecimal string
    """
    if isinstance(messages, str):
        # Hash the string directly
        message_str = messages
    elif isinstance(messages, list):
        # Convert list of dictionaries to JSON string for consistent hashing
        # Sort keys to ensure consistent ordering
        message_str = json.dumps(messages, sort_keys=True, separators=(',', ':'))
    else:
        raise TypeError(f"messages must be str or List[Dict[str, Any]], got {type(messages)}")
    
    # Create SHA-256 hash
    hash_obj = hashlib.sha256(message_str.encode('utf-8'))
    return hash_obj.hexdigest()


def pretty_name(name: str) -> str:
    """
    Convert a name to a pretty name by extracting the last part after '/' and replacing '-' with '_'.
    
    Args:
        name (str): The original model or dataset name/path
        
    Returns:
        str: A cleaned name with last part after '/' and '-' replaced with '_'
    """
    # Extract part after last '/'
    name = name.split('/')[-1]
    # Replace '-' with '_'
    name = name.replace('-', '_')
    return name

def complex_pretty_name(name: str) -> str:
    """
    Convert a name to a pretty name of model name/path or dataset name/path to serve as file name.
    
    This function handles common model/dataset naming conventions like:
    - Hugging Face model names (e.g., "microsoft/DialoGPT-medium")
    - File paths (e.g., "/path/to/model/checkpoint.bin")
    - URLs (e.g., "https://example.com/model.tar.gz")
    - Names with version numbers, special characters, etc.
    
    Args:
        name (str): The original model or dataset name/path
        
    Returns:
        str: A cleaned, file-safe name suitable for use as a filename
    """
    if not name or not isinstance(name, str):
        return "unnamed"
    
    # Start with the original name
    pretty = name.strip()
    
    # Remove URL protocols and domains
    pretty = re.sub(r'^https?://', '', pretty)
    pretty = re.sub(r'^[^/]+\.com/', '', pretty)
    pretty = re.sub(r'^[^/]+\.org/', '', pretty)
    
    # Extract filename from path if it's a full path
    if '/' in pretty:
        # Take the last meaningful part (could be filename or directory name)
        parts = [p for p in pretty.split('/') if p.strip()]
        if parts:
            pretty = parts[-1]
            # If it has a file extension, remove it
            if '.' in pretty and not pretty.startswith('.'):
                pretty = os.path.splitext(pretty)[0]
    
    # Handle common model naming patterns
    # Replace organization separators with underscores
    pretty = pretty.replace('/', '_')
    pretty = pretty.replace('\\', '_')
    
    # Replace common separators and special characters
    pretty = re.sub(r'[-\s]+', '_', pretty)  # Replace hyphens and spaces with underscores
    pretty = re.sub(r'[^\w\-_.]', '_', pretty)  # Replace special chars except word chars, hyphens, underscores, dots
    
    # Clean up multiple underscores
    pretty = re.sub(r'_+', '_', pretty)
    
    # Remove leading/trailing underscores and dots
    pretty = pretty.strip('_.')
    
    # Handle empty result
    if not pretty:
        return "unnamed"
    
    # Ensure it doesn't start with a number (some filesystems don't like this)
    if pretty and pretty[0].isdigit():
        pretty = f"model_{pretty}"
    
    # Truncate if too long (keeping it reasonable for most filesystems)
    max_length = 100
    if len(pretty) > max_length:
        pretty = pretty[:max_length].rstrip('_.')
    
    return pretty



def print_statistics(data: List[dict], output_file: str = None):
    """Print comprehensive statistics about the evaluation results"""
    import sys
    from pathlib import Path
    
    original_stdout = None
    output_file_handle = None
    
    # if output_file is provided, save the statistics to that file by modifying the system stdout
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        output_file_handle = open(output_file, 'w')
        # Save original stdout and redirect to file
        original_stdout = sys.stdout
        sys.stdout = output_file_handle


    pass_rates = []
    test_cases_pass_status = []
    for item in data:
        for eval_result in item['gen_result']['eval_results']:
            pass_rates.append(eval_result['pass_rate'])
            test_cases_pass_status.append(eval_result['test_cases_pass_status'])

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
    pass_rates_per_problem_array = pass_rates_array.reshape(total_problems, -1)
    print(f"\n🎯 Pass Rate Statistics:")
    print(f"   Mean pass rate: {pass_rates_array.mean():.4f} ({pass_rates_array.mean()*100:.2f}%)")
    print(f"   Median pass rate: {np.median(pass_rates_array):.4f} ({np.median(pass_rates_array)*100:.2f}%)")
    print(f"   Std deviation: {pass_rates_array.std():.4f}")
    print(f"   Min pass rate: {pass_rates_array.min():.4f} ({pass_rates_array.min()*100:.2f}%)")
    print(f"   Max pass rate: {pass_rates_array.max():.4f} ({pass_rates_array.max()*100:.2f}%)")

    # print pass@k until solutions, from 1, 2, 4, ..
    print(f"\n📈 Pass@k Statistics:")
    k = 1
    while k < solutions_per_problem:
        pass_at_k = (pass_rates_per_problem_array[:, :k] == 1.0).any(axis=1).mean()
        print(f"   Pass@{k}: {pass_at_k:.4f} ({pass_at_k*100:.2f}%)")
        k *= 2
    pass_at_k = (pass_rates_per_problem_array == 1.0).any(axis=1).mean()
    print(f"   Pass@{solutions_per_problem}: {pass_at_k:.4f} ({pass_at_k*100:.2f}%)")
    print(f"   Total solutions with pass rate 1.0: {np.sum(pass_rates_array == 1.0)}")
    
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
        num_solutions = len(item['gen_result']['eval_results'])
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
    
    # Always restore original stdout and close file handle
    if output_file and original_stdout:
        sys.stdout = original_stdout
    if output_file_handle:
        output_file_handle.close()
        
