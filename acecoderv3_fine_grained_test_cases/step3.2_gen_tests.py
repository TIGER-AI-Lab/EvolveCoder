import os
import datasets
import json
import random
import asyncio
import aiohttp
from collections import defaultdict
from typing import List, Optional, Tuple
from multiprocessing import Pool
from functools import partial
from multiprocessing.pool import ThreadPool
from concurrent.futures import ProcessPoolExecutor, as_completed
from fire import Fire
from tqdm.asyncio import tqdm
from pathlib import Path
from datasets import concatenate_datasets
from evalplus.sanitize import sanitize, code_extract

from acecoderv2.synthesizer.utils import (
    parse_incomplete_json,
    append_jsonl,
    load_jsonl,
    chunking,
    get_python_code_from_string,
    hash_messages,
    pretty_name,
)
from acecoderv2.synthesizer.openai_utils import generate_with_retry, OpenAIAsyncClient

PROMPT_TEMPLATE_RAW = """system:
You are an advanced AI system specialized in generating differentiating test cases that expose logical differences between programs. You will receive a coding prompt and five most similar programs evaluated on the current test cases, along with the test cases and their evaluation results.

Please generate 20 new assert-based test cases that satisfy the following requirements:
- Each test case must clearly differentiate among programs that share the same evaluation results.
    -> At least one program must fail in each test case.
    -> At least one program must pass in each test case.
    -> Test cases should expose different failure modes across the programs.
- All test cases must be correct according to the problem, not based on any specific implementation.
- Use constant values (no randomness or external resource calls).
- Be independent of other test cases.
- Include both input parameters and expected output.

user:
Question: 
{question}

Programs:
```python
{program1}
```
```python
{program2}
```
```python
{program3}
```
```python
{program4}
```
```python
{program5}
```

Existing tests:
{tests}

Evaluation results (rows = programs, cols = tests):
{eval_tests}

Output format (JSON array of strings):
{{"tests": ["assert ...", "assert ..."]}}.
"""

def compute_pass_rates(eval_lists):
    pass_rates = {}
    for i, results in enumerate(eval_lists):
        if not results:
            pass_rates[i] = 0.0
        else:
            pass_rates[i] = sum(results) / len(results)
    return pass_rates


def select_top_two(pass_rates):
    sorted_items = sorted(pass_rates.items(), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in sorted_items[:2]]


def select_candidates_by_pass_rate(pass_rates, exclude_indices, low=0.5, high=0.9):
    return [i for i in pass_rates if i not in exclude_indices and low <= pass_rates[i] <= high]


def pairwise_distance(a, b):
    return sum(x != y for x, y in zip(a, b))


def select_diverse_subset(eval_lists, candidates, k=3):
    if len(candidates) <= k:
        return candidates

    best_triple = None
    max_min_distance = -1
    max_total_distance = -1

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            for l in range(j + 1, len(candidates)):
                a, b, c = candidates[i], candidates[j], candidates[l]
                dist_ab = pairwise_distance(eval_lists[a], eval_lists[b])
                dist_ac = pairwise_distance(eval_lists[a], eval_lists[c])
                dist_bc = pairwise_distance(eval_lists[b], eval_lists[c])

                min_dist = min(dist_ab, dist_ac, dist_bc)
                total_dist = dist_ab + dist_ac + dist_bc

                if (min_dist > max_min_distance or
                    (min_dist == max_min_distance and total_dist > max_total_distance)):
                    max_min_distance = min_dist
                    max_total_distance = total_dist
                    best_triple = (a, b, c)

    return list(best_triple)


def select_programs(eval_lists):
    n = len(eval_lists)
    if n == 0 or len(eval_lists[0]) == 0:
        return random.sample(range(n), min(5, n))

    pass_rates = compute_pass_rates(eval_lists)
    top_two = select_top_two(pass_rates)
    candidates = select_candidates_by_pass_rate(pass_rates, exclude_indices=top_two)

    if len(candidates) >= 3:
        selected = select_diverse_subset(eval_lists, candidates, k=3)
    else:
        extra = [i for i in pass_rates if i not in top_two and pass_rates[i] > 0.1]
        selected = random.sample(extra, min(3, len(extra)))

    return sorted(top_two + selected)


def extract_eval_lists(item, tests):
    eval_lists = []
    for eval_result in item['gen_result']['eval_results']:
        pass_status = [status['pass'] for status in eval_result['test_cases_pass_status']]
        assert len(pass_status) == len(tests), (
            f"len(pass_status) {len(pass_status)} != len(tests) {len(tests)}"
        )
        eval_lists.append(pass_status)
    assert eval_lists, "eval_lists should not be empty"
    return eval_lists


def select_eval_indices(eval_lists):
    eval_index = select_programs(eval_lists)
    assert len(eval_index) == 5, f"len(eval_index) {len(eval_index)} != 5"
    return eval_index


def build_eval_matrix_and_solutions(item, eval_lists, eval_index, tests):
    eval_matrix = []
    solutions = []
    for idx in eval_index:
        assert len(eval_lists[idx]) == len(tests), (
            f"len(eval_lists[{idx}]) {len(eval_lists[idx])} != len(tests) {len(tests)}"
        )
        eval_matrix.append(eval_lists[idx])
        solutions.append(item['gen_result']['eval_results'][idx]['parse_code'])
    return eval_matrix, solutions


def format_eval_matrix(eval_matrix):
    return "[\n" + "\n".join(f"    {row}," for row in eval_matrix) + "\n]"


def process_item(item, idx):
    problem = item['problem']
    tests = item['filtered_tests']
    assert len(item['outputs']) == len(item['gen_result']['eval_results']), (
        f"outputs({len(item['outputs'])}) != eval_results({len(item['gen_result']['eval_results'])})"
    )

    eval_lists = extract_eval_lists(item, tests)

    eval_index = select_eval_indices(eval_lists)

    eval_matrix, solutions = build_eval_matrix_and_solutions(item, eval_lists, eval_index, tests)

    eval_matrix_str = format_eval_matrix(eval_matrix)

    return {
        "id": item["id"],
        "problem": problem,
        "raw_tests": item["raw_tests"],
        "outputs": item["outputs"],
        "filtered_tests": tests,
        "sampled_solutions": solutions,
        "eval_matrix": eval_matrix_str,
    }

def preprocess_dataset(file_path: str, max_sample=None, num_proc=4) -> datasets.Dataset:
    """
    Preprocess the dataset to extract the relevant fields for test case generation.
    Args:
        dataset_name (str): The name of the dataset to preprocess.
        max_sample (int, optional): Maximum number of samples to process. Defaults to None.
        num_proc (int, optional): Number of processes to use for preprocessing. Defaults to 4.
    Returns:
        datasets.Dataset: The preprocessed dataset. each item with keys:
            - "id": unique identifier for the item
            - "problem": the instruction for the code generation
            - "response": the raw response as the solution, combined by natural language and code
            - "program": the code snippet that answers the question
    """ 
    dataset = datasets.Dataset.from_json(file_path)
    dataset = dataset.select(range(max_sample))
        
    dataset = dataset.map(
        process_item,
        with_indices=True,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
    )
    return dataset


async def process_batch_async(
    client: OpenAIAsyncClient,
    batch_items: List[dict],
    model_name: str,
    max_tokens: int,
    cache_file: Path,
    max_concurrent: int = 10,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    **kwargs
) -> List[dict]:
    """
    Process a batch of items asynchronously.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_single_item(item):
        # Prepare prompt
        prompt = PROMPT_TEMPLATE_RAW.format(
            question=item['problem'],
            tests=item['filtered_tests'],
            program1=item['sampled_solutions'][0],
            program2=item['sampled_solutions'][1],
            program3=item['sampled_solutions'][2],
            program4=item['sampled_solutions'][3],
            eval_tests=item['eval_matrix']
        )
        
        messages = [{"role": "user", "content": prompt}]
        
        # Generate new response
        response = await generate_with_retry(
            client=client,
            messages=messages,
            model=model_name,
            max_tokens=max_tokens,
            max_retries=max_retries,
            retry_delay=retry_delay,
            semaphore=semaphore,
            **kwargs
        )
        
        # Update item with response
        result_item = item.copy()
        result_item['synthesis_result']['gpt_prompt'] = prompt
        result_item['synthesis_result']['gpt_response'] = response[0]
        return result_item
    
    # Process all items in the batch concurrently
    tasks = [process_single_item(item) for item in batch_items]
    results = await tqdm.gather(*tasks, desc="Processing batch items", total=len(tasks))
    append_jsonl(cache_file, [item['synthesis_result'] for item in results])
    return results

FILE_NAME = Path(__file__).stem
default_output_dir = Path(__file__).parent / "outputs"
async def main_async(
    file_path: str,
    max_samples: Optional[int] = None,
    model_name: str = "o3-mini-2025-01-31",
    round: int = 0,
    max_tokens: int = 8192,
    top_p: float = 0.95,
    temperature: float = 0.6,
    seed: int = 42,
    n: int = 1,
    num_proc: int = 4,
    output_dir: str = default_output_dir,
    overwrite: bool = False,
    save_batch_size: int = 20,
    max_concurrent: int = 10,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    batch_delay: float = 0.5,
    api_key: Optional[str] = None,
    base_url: str = "https://api.openai.com/v1",
):
    """
    Main async function to generate test cases for a given dataset.
    """
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
    
    print(f"Generating round: {round}")
    print(f"Processing file: {file_path}")
    print(f"Model: {model_name}")
    print(f"Max samples: {max_samples}")
    print(f"Batch size: {save_batch_size}")
    print(f"Max concurrent: {max_concurrent}")
    
    # Setup paths
    output_dir = Path(output_dir) if output_dir else Path(file_path).parent
    cache_file = Path(output_dir) / f"{FILE_NAME}_round_{round}.cache.jsonl"
    output_file = output_dir / f"{FILE_NAME}_results_round_{round}.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if output_file.exists() and not overwrite:
        print(f"Output file {output_file} already exists. Use --overwrite to overwrite.")
        return

    # Preprocess dataset
    dataset = preprocess_dataset(file_path, max_sample=max_samples, num_proc=num_proc)
    data = list(dataset)  # Convert to list for easier processing

    # Load existing cache
    cached_data = {}
    if cache_file.exists():
        print(f"Loading existing cache from {cache_file}")
        existing_cache = load_jsonl(cache_file)
        cached_data = {item['hash_id']: item for item in existing_cache}
        print(f"Loaded {len(cached_data)} cached items")
    
    # Identify items that need processing
    items_to_process = []
    items_to_process_map = {}
    final_results = []
    
    for i, item in enumerate(data):
        # Prepare messages to get hash_id
        prompt = PROMPT_TEMPLATE_RAW.format(
            question=item['problem'],
            tests=item['filtered_tests'],
            program1=item['sampled_solutions'][0],
            program2=item['sampled_solutions'][1],
            program3=item['sampled_solutions'][2],
            program4=item['sampled_solutions'][3],
            program5=item['sampled_solutions'][4],
            eval_tests=item['eval_matrix']
        )
            
        messages = [{"role": "user", "content": prompt}]
        hash_id = hash_messages(messages)
        item['synthesis_result'] = {
            "hash_id": hash_id,
        }
        if hash_id in cached_data:
            item['synthesis_result']['gpt_response'] = cached_data[hash_id]['gpt_response']
        else:
            items_to_process_map[hash_id] = i
            items_to_process.append(item)
        final_results.append(item)  # Initialize with current item
    
    print(f"Found {len(cached_data)} cached items, {len(items_to_process)} items need processing")
    
    if len(items_to_process) == 0:
        print("All items are cached, saving final results...")
        # Save final results
        with open(output_file, 'w') as f:
            for item in final_results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Results saved to {output_file}")
        return
    
    # Process items in batches using async client
    async with OpenAIAsyncClient(api_key=api_key, base_url=base_url) as client:
        num_batches = (len(items_to_process) + save_batch_size - 1) // save_batch_size
        processed_count = 0
        
        for i in tqdm(range(0, len(items_to_process), save_batch_size), desc="Processing batches"):
            batch_items = items_to_process[i:i + save_batch_size]
            
            # Process batch
            batch_results = await process_batch_async(
                client=client,
                batch_items=batch_items,
                model_name=model_name,
                max_tokens=max_tokens,
                cache_file=cache_file,
                max_concurrent=max_concurrent,
                max_retries=max_retries,
                retry_delay=retry_delay,
                temperature=temperature,
                top_p=top_p,
                n=n,
                seed=seed,
            )
            
            # Update final results
            for j, result_item in enumerate(batch_results):
                hash_id = result_item['synthesis_result']['hash_id']
                idx = items_to_process_map[hash_id]
                final_results[idx] = result_item
            
            processed_count += len(batch_results)
            print(f"Processed batch {i//save_batch_size + 1}/{num_batches} ({processed_count}/{len(items_to_process)} items)")
            
            # Add delay between batches
            if batch_delay > 0 and i + save_batch_size < len(items_to_process):
                await asyncio.sleep(batch_delay)
    
    # Save final results
    with open(output_file, 'w') as f:
        for item in final_results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Processed dataset saved to {output_file}")
    
    # Remove cache file
    if cache_file.exists():
        os.remove(cache_file)
        print(f"Cache file {cache_file} removed")


def main(
    file_path: str,
    max_samples: Optional[int] = None,
    model_name: str = "o3-mini-2025-01-31",
    round: int = 0,
    max_tokens: int = 8192,
    top_p: float = 0.95,
    temperature: float = 0.6,
    seed: int = 42,
    n: int = 1,
    num_proc: int = 4,
    output_dir: str = None,
    overwrite: bool = False,
    save_batch_size: int = 20,
    max_concurrent: int = 10,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    batch_delay: float = 0.5,
    api_key: Optional[str] = None,
    base_url: str = "https://api.openai.com/v1",
):
    """
    Synchronous wrapper for the async main function.
    """
    try:
        asyncio.run(main_async(
            file_path=file_path,
            max_samples=max_samples,
            model_name=model_name,
            round=round,
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
            seed=seed,
            n=n,
            num_proc=num_proc,
            output_dir=output_dir,
            overwrite=overwrite,
            save_batch_size=save_batch_size,
            max_concurrent=max_concurrent,
            max_retries=max_retries,
            retry_delay=retry_delay,
            batch_delay=batch_delay,
            api_key=api_key,
            base_url=base_url,
        ))
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    Fire(main)

"""
python acecoderv3_fine_grained_test_cases/step3.2_gen_tests.py acecoderv3_fine_grained_test_cases/outputs/all_20_round1/gpt_4.1_mini/step3.1_filter_tests_round_1.jsonl --round 1 --max_samples 2 --model_name gpt-4.1-mini --save_batch_size 1 --max_concurrent 1 --batch_delay 2.0
"""