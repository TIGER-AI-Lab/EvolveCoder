import os
import datasets
import json
import random
import asyncio
import aiohttp
from typing import List, Optional, Tuple
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
You are a highly capable AI system specialized in stress-testing algorithms and generating edge test cases for competitive programming and LeetCode-style problems.

You will be provided with:
1. A hard-level algorithmic question (LeetCode style).
2. A list of existing assert-based test cases.
3. Multiple code solutions that attempt to solve the question.
4. The evaluation results of each solution on those existing tests.

Your task:
1. Analyze the question carefully to understand the full range of input constraints and edge conditions.
2. Identify weaknesses or potential failure points among the solutions:
    - Logical differences that could lead to diverging outputs.
    - Boundary conditions that some solutions may not handle.
    - Performance bottlenecks or recursion/overflow risks.
3. Design **20 upgraded assert-based test cases** that:
    - Stress-test all edge conditions (min/max input sizes, empty structures, uniform values, pathological patterns).
    - Include both small tricky inputs and large complex stress tests.
    - Are deterministic and self-contained.
    - Aim to expose potential discrepancies among solutions.
4. If possible, make each case capable of distinguishing between correct and incorrect logic among the given solutions.
5. Keep the same assert format as before: `assert func(input) == expected_output`
6. All outputs must be correct according to the problem definition, not any specific implementation.


user:
Question: 
{question}

Reference solutions:
```python
{program1}
```
```python
{program2}
```
```python
{program3}
```

Existing tests:
{tests}

Evaluation results (rows = tests, cols = [p1,p2,p3]):
{eval_tests}

Output format (JSON array of strings):
{{"tests": ["assert ...", "assert ..."]}}.
"""


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
        
    def process_item(item, idx):
        problem = item['problem']
        tests = item['filtered_tests']
        
        pass_rates = [entry['pass_rate'] for entry in item['gen_result']['eval_results']]
        eval_results = [entry for entry in item['gen_result']['eval_results']]
        assert len(pass_rates) == len(eval_results), f"len(pass_rates) == len(eval_results), {len(pass_rates)} == {len(eval_results)}"
        
        top_k = 3
        threshold = 0.6
        filtered = [(r, e) for r, e in zip(pass_rates, eval_results) if r > threshold]
        if len(filtered) >= top_k:
            selected = random.sample(filtered, top_k)
        else:
            selected = sorted(zip(pass_rates, eval_results), key=lambda x: x[0], reverse=True)
        solutions = []
        pass_matrix = []
        for _, eval_result in selected:
            solutions.append(code_extract(eval_result['parse_code']))
            pass_status = []
            for status in eval_result['test_cases_pass_status']:
                pass_status.append(status['pass'])
            assert len(pass_status) == len(tests), f"len(pass_status) {len(pass_status)} == len(tests) {len(tests)}"
            pass_matrix.append(pass_status)
        
        return {
            "id": item['id'],
            "problem": problem,
            "raw_tests": item['raw_tests'],
            "outputs": item['outputs'],
            "filtered_tests": tests,
            "sampled_solutions": solutions,
            "eval_matrix": pass_matrix
        }
        
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
    round: int,
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
                round=round,
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
python acecoderv3/step3.2_gen_tests.py acecoderv3/outputs/all_20/gpt_4.1_mini/step3.1_filter_tests_round_1.jsonl --round 1 --max_samples 2 --model_name gpt-4.1-mini --save_batch_size 1 --max_concurrent 1 --batch_delay 2.0
"""