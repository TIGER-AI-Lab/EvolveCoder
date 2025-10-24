import fire
import json
import datasets
import random
from acecoderv3_fine_grained_test_cases.code_eval import eval_codes, parse_code
from acecoderv3_fine_grained_test_cases.utils import print_statistics
from typing import List, Union, Optional
from pathlib import Path
import numpy as np
from collections import Counter
import signal
import sys
import os
from contextlib import contextmanager

FILE_NAME = Path(__file__).stem
LAST_STEP_NAME = "step3.3_parsing_tests"

class TimeoutException(Exception):
    pass

def restart_program():
    """Restart the current program with the same arguments"""
    print(f"\n🔄 Restarting program...")
    print(f"   Command: {' '.join(sys.argv)}")
    python = sys.executable
    os.execl(python, python, *sys.argv)

@contextmanager
def timeout(seconds):
    """Context manager for timeout using signal alarm"""
    def timeout_handler(signum, frame):
        raise TimeoutException(f"Batch processing exceeded timeout of {seconds} seconds")
    
    # Set the signal handler and alarm
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Restore the old handler and cancel alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def load_cache(cache_file: Path):
    """Load processed items from cache file"""
    if not cache_file.exists():
        return {}
    
    cache = {}
    print(f"📂 Loading cache from: {cache_file}")
    with open(cache_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            item_id = item.get('id')
            if item_id:
                cache[item_id] = item
    
    print(f"✅ Loaded {len(cache)} cached items")
    return cache

def save_to_cache(item, cache_file: Path):
    """Append a single item to cache file"""
    with open(cache_file, 'a') as f:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main(
    file_path: str,
    output_dir: str = None,
    overwrite: bool = False,
    num_proc: int = 64,
    round: int = 1,
    max_samples: Optional[int] = 0,
    batch_size: int = 100,
    batch_timeout: int = 1200,  # Default 25 minutes timeout per batch
    auto_restart: bool = True  # Auto restart on timeout
):
    output_dir = Path(output_dir) if output_dir else Path(file_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    print(FILE_NAME)
    output_file = output_dir / f"{FILE_NAME}_round_{round}.jsonl"
    cache_file = output_dir / f"{FILE_NAME}_round_{round}.cache.jsonl"
    stats_output_file = output_dir / f"{FILE_NAME}_stats_round_{round}.txt"
    
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
    
    # Ensure all items have IDs
    for i, item in enumerate(data):
        if 'id' not in item:
            item['id'] = f"item_{i}"
    
    if max_samples > 0 and len(data) > max_samples:
        random.seed(42)  # For reproducibility
        random_idxs = set(random.sample(range(len(data)), max_samples))
        data = [data[i] for i in range(len(data)) if i in random_idxs]

    print(f"📥 Loaded {len(data)} problems")

    # Load cache and filter out processed items
    if overwrite and cache_file.exists():
        cache_file.unlink()
        print(f"🗑️  Removed existing cache file")
    
    cache = load_cache(cache_file)
    
    # Separate processed and unprocessed items
    processed_data = []
    unprocessed_data = []
    
    for item in data:
        item_id = item['id']
        if item_id in cache:
            processed_data.append(cache[item_id])
        else:
            unprocessed_data.append(item)
    
    print(f"✅ Already processed: {len(processed_data)} items")
    print(f"⏳ To process: {len(unprocessed_data)} items")

    if len(unprocessed_data) == 0:
        print("🎉 All items already processed!")
        all_data = processed_data
    else:
        # Process in batches
        newly_processed = []
        for batch_start in range(0, len(unprocessed_data), batch_size):
            batch_end = min(batch_start + batch_size, len(unprocessed_data))
            batch_data = unprocessed_data[batch_start:batch_end]
            
            print(f"\n🔧 Processing batch {batch_start//batch_size + 1}/{(len(unprocessed_data)-1)//batch_size + 1} "
                  f"(items {batch_start+1}-{batch_end}/{len(unprocessed_data)})...")
            print(f"⏱️  Batch timeout: {batch_timeout} seconds")
            
            try:
                with timeout(batch_timeout):
                    # Extract solutions and test cases for this batch
                    solution_strs = []
                    test_cases = []
                    item_indices = []
                    
                    for item_idx, item in enumerate(batch_data):
                        eval_tests = item['raw_tests'] + item['synthesis_result']['tests']
                        for output in item['outputs']:
                            solution_strs.append(output)
                            test_cases.append(eval_tests)
                            item_indices.append(item_idx)

                    print(f"   Processing {len(solution_strs)} solutions...")

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

                    print("   ⚡ Evaluating codes...")
                    pass_rates, test_cases_info = eval_codes(
                        solution_strs=dataset['parse_code'],
                        test_cases=dataset['test_case'],
                        return_test_cases_pass_status=True,
                        binary=False,
                        num_processes=num_proc,
                    )
                    
                    test_cases_pass_status = [info['details'] for info in test_cases_info]

                    # Reassign results back to batch items
                    idx = 0
                    for item in batch_data:
                        item['tests'] = item['raw_tests'] + item['synthesis_result']['tests']
                        item.pop('filtered_tests', None)
                        item.pop('raw_tests', None)
                        item['gen_result'] = {}
                        item['gen_result']['eval_results'] = []
                        test_case_diversity_arr = []
                        for _ in range(len(item['outputs'])):
                            item['gen_result']['eval_results'].append({
                                'pass_rate': pass_rates[idx],
                                'test_cases_pass_status': test_cases_pass_status[idx],
                                'parse_code': dataset['parse_code'][idx]
                            })
                            test_case_diversity_arr.append([x['pass'] for x in test_cases_pass_status[idx]])
                            idx += 1
                        test_case_diversity_arr = np.array(test_case_diversity_arr).T.tolist()
                        item['gen_result']['test_case_diversity'] = {
                            "arr": test_case_diversity_arr,
                            "mean": np.mean(test_case_diversity_arr, axis=1).tolist(),
                        }
                        
                        # Save to cache immediately
                        save_to_cache(item, cache_file)
                        newly_processed.append(item)
                    
                    print(f"   ✅ Batch {batch_start//batch_size + 1} completed and cached")
                    
            except TimeoutException as e:
                print(f"\n⚠️  {e}")
                print(f"💾 Progress saved to cache: {len(newly_processed)} items processed in this session")
                print(f"   Remaining items: {len(unprocessed_data) - batch_end}")
                
                if auto_restart:
                    print(f"🔄 Auto-restarting in 3 seconds...")
                    import time
                    time.sleep(3)
                    restart_program()
                else:
                    print(f"🔄 Please restart the program to continue processing")
                    sys.exit(1)
            except (KeyboardInterrupt, SystemExit):
                # Handle user interruption or system exit
                print(f"\n\n⚠️  Process interrupted")
                print(f"💾 Progress saved to cache: {len(newly_processed)} items processed in this session")
                print(f"🔄 You can restart the program to continue processing")
                raise
            except Exception as e:
                # Check if this is a timeout-related error from multiprocessing
                error_msg = str(e).lower()
                error_type = type(e).__name__
                
                is_timeout = (
                    'timeout' in error_msg or 
                    error_type in ('TimeoutError', 'TimeoutException') or
                    'exceeded timeout' in error_msg
                )
                
                if is_timeout:
                    print(f"\n⚠️  Batch processing timeout detected: {e}")
                    print(f"💾 Progress saved to cache: {len(newly_processed)} items processed in this session")
                    print(f"   Remaining items: {len(unprocessed_data) - batch_end}")
                    
                    if auto_restart:
                        print(f"🔄 Auto-restarting in 3 seconds...")
                        import time
                        time.sleep(3)
                        restart_program()
                    else:
                        print(f"🔄 Please restart the program to continue processing")
                        sys.exit(1)
                else:
                    print(f"\n❌ Error processing batch: {e}")
                    print(f"💾 Progress saved to cache: {len(newly_processed)} items processed so far")
                    print(f"🔄 You can restart the program to continue processing")
                    raise
        
        all_data = processed_data + newly_processed

    # Save final output
    print(f"\n💾 Saving final results to: {output_file}")
    with open(output_file, 'w') as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Print comprehensive statistics
    print_statistics(all_data, output_file=stats_output_file)
    print(f"✅ Results saved to {output_file}")
    
    
    
if __name__ == "__main__":
    fire.Fire(main)

"""
# Basic usage
python acecoderv3_fine_grained_test_cases/step3.4_eval_combined_tests.py acecoderv3_fine_grained_test_cases/outputs/all_20_round1/gpt_4.1_mini/step3.3_parsing_tests_round_1.jsonl --round 1

# With custom batch size and timeout (auto-restart enabled by default)
python acecoderv3_fine_grained_test_cases/step3.4_eval_combined_tests.py acecoderv3_fine_grained_test_cases/outputs/all_20_round1/gpt_4.1_mini/step3.3_parsing_tests_round_1.jsonl --round 1 --batch_size=50 --batch_timeout=1800

# Disable auto-restart (manual restart required)
python acecoderv3_fine_grained_test_cases/step3.4_eval_combined_tests.py input.jsonl --round 1 --auto_restart=False

# Force overwrite (clear cache)
python acecoderv3_fine_grained_test_cases/step3.4_eval_combined_tests.py acecoderv3_fine_grained_test_cases/outputs/all_20_round1/gpt_4.1_mini/step3.3_parsing_tests_round_1.jsonl --round 1 --overwrite=True

# Example with 30 minute timeout per batch and auto-restart
python acecoderv3_fine_grained_test_cases/step3.4_eval_combined_tests.py input.jsonl --round 1 --batch_timeout=1800 --auto_restart=True
"""