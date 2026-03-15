import os
import datasets
import json
import asyncio
import aiohttp
from typing import List, Optional, Tuple
from fire import Fire
from tqdm.asyncio import tqdm
from pathlib import Path
from datasets import concatenate_datasets

from evolvecoder.utils import (
    parse_incomplete_json,
    append_jsonl,
    load_jsonl,
    chunking,
    get_python_code_from_string,
    hash_messages,
    pretty_name,
)
from evolvecoder.openai_utils import generate_with_retry, OpenAIAsyncClient

PROMPT_TEMPLATE_RAW = """system:
You are the latest and best bot aimed at transforming some code snippet into a very challenging LeetCode-style question intended for advanced CS university students and experienced software engineers. You will be provided with a prompt for writing code, along with a reference program that attempts to answer the question. Please complete the following for me:
1. Create a LeetCode-style question that meets these requirements:
    - The question must be hard or very hard difficulty level (similar to the hardest LeetCode problems).
    - The question should require advanced algorithmic thinking, such as:
        -> Graph theory with dynamic programming.
        -> Advanced string processing (suffix arrays, KMP, etc.).
        -> Complex greedy + data structure combinations.
        -> Sliding windows with optimization, interval DP, or segment trees.
    - The question must:
        -> Contain a function signature, rather than stdin/stdout style.
        -> Be self-contained (no external resources or data).
        -> Be challenging enough that solving it takes 30–60 minutes for experts.
        -> Avoid machine learning, OS-level concepts, or anything requiring system calls or file I/O.
    - Do NOT request time/space complexity analysis or ask for test cases in the question text.
    - You can take inspiration from the reference code snippet, but you may discard parts of it if necessary to make the question cleaner and harder.
2. Based on the question you create:
    - Generate 20 independent test cases using assert statements.
    - Each test case must:
        -> Use constant values (no randomness or external resource calls).
        -> Be independent of other test cases.
        -> Include both input parameters and expected output.
        
user:
Here is the original question:
{instruction}

Here is the reference program that answers the question:
```python
{program}
```

Now give your modified question and generated test cases in the following json format: 
{{"question": ..., "tests":["assert ...", "assert ..."]}}.
"""

PROMPT_TEMPLATE_NO_SOLUTION = """system:
You are the latest and best bot aimed at transforming some code snippet into a very challenging LeetCode-style question intended for advanced CS university students and experienced software engineers. You will be provided with a prompt for writing code. Please complete the following for me:
1. Create a LeetCode-style question that meets these requirements:
    - The question must be hard or very hard difficulty level (similar to the hardest LeetCode problems).
    - The question should require advanced algorithmic thinking, such as:
        -> Graph theory with dynamic programming.
        -> Advanced string processing (suffix arrays, KMP, etc.).
        -> Complex greedy + data structure combinations.
        -> Sliding windows with optimization, interval DP, or segment trees.
    - The question must:
        -> Contain a function signature, rather than stdin/stdout style.
        -> Be self-contained (no external resources or data).
        -> Be challenging enough that solving it takes 30–60 minutes for experts.
        -> Avoid machine learning, OS-level concepts, or anything requiring system calls or file I/O.
    - Do NOT request time/space complexity analysis or ask for test cases in the question text.
    - You can take inspiration from the reference code snippet, but you may discard parts of it if necessary to make the question cleaner and harder.
2. Based on the question you create:
    - Generate 20 independent test cases using assert statements.
    - Each test case must:
        -> Use constant values (no randomness or external resource calls).
        -> Be independent of other test cases.
        -> Include both input parameters and expected output.
        
user:
Here is the original question:
{instruction}

Now give your modified question and generated test cases in the following json format: 
{{"question": ..., "tests":["assert ...", "assert ..."]}}.
"""

def remove_code_wrapper(solution):
    if not isinstance(solution, str):
        return solution
    if solution.startswith('```python'):
        solution = solution[9:].lstrip('\n')
        if solution.endswith('```'):
            solution = solution[:-3].rstrip('\n')
    elif solution.startswith('```'):
        solution = solution[3:].lstrip('\n')
        if solution.endswith('```'):
            solution = solution[:-3].rstrip('\n')
    
    return solution

def preprocess_dataset(sub_dataset_name: str, max_sample=None, num_proc=4) -> datasets.Dataset:
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
    data = "CodeDPO/filtered_original_acecoderv3_unused"
    if sub_dataset_name == 'all':
        sub_dataset_names = ["TACO", "APPS", "primeintellect", "codeforces", "contests"]
        all_datasets = []
        if max_sample is not None:
            samples_per_subset = max_sample // len(sub_dataset_names)
            remainder = max_sample % len(sub_dataset_names)      
            for i, name in enumerate(sub_dataset_names):
                sub_ds = datasets.load_dataset(data, name, split="train")
                num_samples = samples_per_subset + (1 if i < remainder else 0)
                num_samples = min(num_samples, len(sub_ds))
                sub_ds = sub_ds.select(range(num_samples))
                all_datasets.append(sub_ds)
        else:
            for name in sub_dataset_names:
                sub_ds = datasets.load_dataset(data, name, split="train")
                all_datasets.append(sub_ds)
        dataset = concatenate_datasets(all_datasets)
    else:
        dataset = datasets.load_dataset(data, sub_dataset_name, split="train")
        if max_sample is not None:
            dataset = dataset.select(range(max_sample))
        
    def process_item(item, idx):
        problem = item.get("question")
        starter_code = item.get('starter_code')
        if starter_code:
            problem += f"\n\nHere is the starter code:\n{starter_code}"
        solution = item.get("solution")
        id = item.get("id")
        id_prefix = id.split('_')[0]
        if id_prefix == "primeintellect":
            solution = remove_code_wrapper(solution)
        return {
            "id": id,
            "problem": problem,
            "program": solution
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
        if item['program'] is not None:
            prompt = PROMPT_TEMPLATE_RAW.format(
                instruction=item['problem'],
                program=item['program']
            )
        else:
            prompt = PROMPT_TEMPLATE_NO_SOLUTION.format(
                instruction=item['problem'],
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
    sub_dataset_name: str = "TACO",
    max_samples: Optional[int] = None,
    model_name: str = "o3-mini-2025-01-31",
    max_tokens: int = 8192,
    top_p: float = 0.95,
    temperature: float = 0.6,
    seed: int = 42,
    n: int = 1,
    num_proc: int = 16,
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
    
    print(f"Processing sub dataset: {sub_dataset_name}")
    print(f"Model: {model_name}")
    print(f"Max samples: {max_samples}")
    print(f"Batch size: {save_batch_size}")
    print(f"Max concurrent: {max_concurrent}")
    
    # Setup paths
    output_dir = Path(output_dir) if output_dir else default_output_dir
    output_dir = Path(output_dir) / pretty_name(sub_dataset_name) / pretty_name(model_name)
    cache_file = Path(output_dir) / f"{FILE_NAME}.cache.jsonl"
    output_file = output_dir / f"{FILE_NAME}_results.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if output_file.exists() and not overwrite:
        print(f"Output file {output_file} already exists. Use --overwrite to overwrite.")
        return

    # Preprocess dataset
    dataset = preprocess_dataset(sub_dataset_name, max_sample=max_samples, num_proc=num_proc)
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
        if item['program'] is not None:
            prompt = PROMPT_TEMPLATE_RAW.format(
                instruction=item['problem'],
                program=item['program']
            )
        else:
            prompt = PROMPT_TEMPLATE_NO_SOLUTION.format(
                instruction=item['problem'],
            )
        
        messages = [{"role": "user", "content": prompt}]
        hash_id = hash_messages(messages)
        item['synthesis_result'] = {
            "hash_id": hash_id,
        }
        if hash_id in cached_data:
            item['synthesis_result']['gpt_response'] = cached_data[hash_id]['gpt_response']
            item['synthesis_result']['gpt_prompt'] = cached_data[hash_id]['gpt_prompt']
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
    sub_dataset_name: str = "TACO",
    max_samples: Optional[int] = None,
    model_name: str = "o3-mini-2025-01-31",
    max_tokens: int = 8192,
    top_p: float = 0.95,
    temperature: float = 0.6,
    seed: int = 42,
    n: int = 1,
    num_proc: int = 16,
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
    Synchronous wrapper for the async main function.
    """
    try:
        asyncio.run(main_async(
            sub_dataset_name=sub_dataset_name,
            max_samples=max_samples,
            model_name=model_name,
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
python evolvecoder/step1_prompting.py --sub_dataset_name APPS --max_samples 2 --model_name gpt-4.1-mini --save_batch_size 1 --max_concurrent 1 --batch_delay 2.0
"""