import os
import datasets
import json
import asyncio
import aiohttp
from typing import List, Optional, Tuple
from fire import Fire
from tqdm.asyncio import tqdm
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
from acecoderv2.synthesizer.openai_utils import generate_with_retry, OpenAIAsyncClient

PROMPT_TEMPLATE_RAW = """system:
You are the latest and best bot aimed at transforming some code snippet into a very challenging LeetCode-style question intended for advanced CS university students and experienced software engineers. You will be provided with a prompt for writing code, along with a reference program that attempts to answer the question. Please complete the following for me:
1. Create a LeetCode-style question that meets these requirements:
    - The question must be hard or very hard difficulty level (similar to the hardest LeetCode problems).
    - The problem should require advanced algorithmic thinking, such as:
        -> Graph theory with dynamic programming.
        -> Advanced string processing (suffix arrays, KMP, etc.).
        -> Complex greedy + data structure combinations.
        -> Sliding windows with optimization, interval DP, or segment trees.
    - The question should have a clear, precise statement, including:
        -> Input description.
        -> Output description.
        -> Example inputs and outputs with explanations.
    - The question must:
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

PROMPT_TEMPLATE_NO_INSTRUCTION = """system:
You are the latest and best bot aimed at transforming some code snippet into a very challenging LeetCode-style question intended for advanced CS university students and experienced software engineers. You will be provided with a reference program that attempts to answer the question. Please complete the following for me:
1. Create a LeetCode-style question that meets these requirements:
    - The question must be hard or very hard difficulty level (similar to the hardest LeetCode problems).
    - The problem should require advanced algorithmic thinking, such as:
        -> Graph theory with dynamic programming.
        -> Advanced string processing (suffix arrays, KMP, etc.).
        -> Complex greedy + data structure combinations.
        -> Sliding windows with optimization, interval DP, or segment trees.
    - The question should have a clear, precise statement, including:
        -> Input description.
        -> Output description.
        -> Example inputs and outputs with explanations.
    - The question must:
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
Here is the reference program:
```python
{program}
```

Now give your modified question and generated test cases in the following json format: 
{{"question": ..., "tests":["assert ...", "assert ..."]}}.
"""


def preprocess_dataset(dataset_name: str, max_sample=None, num_proc=4) -> datasets.Dataset:
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
    if dataset_name == "ise-uiuc/Magicoder-Evol-Instruct-110K":
        dataset = datasets.load_dataset(
            dataset_name,
            split="train",
        )
        if max_sample is not None and max_sample > 0:
            dataset = dataset.select(range(max_sample))
        
        def process_item(item, idx):
            return {
                "id": idx,
                "problem": item.get("instruction"),
                "response": item.get("response"),
                "program": get_python_code_from_string(item['response']) if 'response' in item else None
            }
        dataset = dataset.map(
            process_item,
            with_indices=True,
            remove_columns=dataset.column_names,
            num_proc=num_proc,
        )
        return dataset
    if dataset_name == "ise-uiuc/Magicoder-OSS-Instruct-75K":
        dataset = datasets.load_dataset(
            dataset_name,
            split="train",
        )
        # only keep python
        dataset = dataset.filter(lambda x: x['lang'] == 'python')
        if max_sample is not None:
            dataset = dataset.select(range(max_sample))
        def process_item(item, idx):
            return {
                "id": item['index'],
                "problem": item.get("problem"),
                "response": item.get("solution"),
                "program": get_python_code_from_string(item['solution']) if 'solution' in item else None
            }
        dataset = dataset.map(
            process_item,
            with_indices=True,
            remove_columns=dataset.column_names,
            num_proc=num_proc,
        )
        return dataset
    if dataset_name == "bigcode/stack-dedup-python-fns":
        dataset = datasets.load_dataset(
            dataset_name,
            split="train",
        )
        if max_sample is not None:
            dataset = dataset.select(range(max_sample))
        def process_item(item, idx):
            return {
                "id": item['id'],
                "problem": None,
                "response": None,
                "program": item['content']
            }
        dataset = dataset.map(
            process_item,
            with_indices=True,
            remove_columns=dataset.column_names,
            num_proc=num_proc,
        )
        return dataset
    raise ValueError(f"Dataset {dataset_name} is not supported for preprocessing.")



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
        if item['problem'] is not None:
            prompt = PROMPT_TEMPLATE_RAW.format(
                program=item['program'],
                instruction=item['problem']
            )
        else:
            prompt = PROMPT_TEMPLATE_NO_INSTRUCTION.format(
                program=item['program']
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
    dataset_name: str = "ise-uiuc/Magicoder-Evol-Instruct-110K",
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
    
    print(f"Processing dataset: {dataset_name}")
    print(f"Model: {model_name}")
    print(f"Max samples: {max_samples}")
    print(f"Batch size: {save_batch_size}")
    print(f"Max concurrent: {max_concurrent}")
    
    # Preprocess dataset
    dataset = preprocess_dataset(dataset_name, max_sample=max_samples, num_proc=num_proc)
    data = list(dataset)  # Convert to list for easier processing
    
    # Setup paths
    output_dir = Path(output_dir) if output_dir else default_output_dir
    output_dir = Path(output_dir) / pretty_name(dataset_name) / pretty_name(model_name)
    cache_file = Path(output_dir) / f"{FILE_NAME}.cache.jsonl"
    output_file = output_dir / f"{FILE_NAME}_results.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if output_file.exists() and not overwrite:
        print(f"Output file {output_file} already exists. Use --overwrite to overwrite.")
        return

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
        if item['problem'] is not None:
            prompt = PROMPT_TEMPLATE_RAW.format(
                program=item['program'],
                instruction=item['problem']
            )
        else:
            prompt = PROMPT_TEMPLATE_NO_INSTRUCTION.format(
                program=item['program']
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
    dataset_name: str = "ise-uiuc/Magicoder-Evol-Instruct-110K",
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
            dataset_name=dataset_name,
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
This code is part of the AceCoderV2 project, which is designed to generate challenging LeetCode-style questions and test cases from code snippets using OpenAI's GPT models. The main function orchestrates the preprocessing of datasets, generation of test cases, and saving the results to a specified output directory. It supports async processing for efficiency and allows for caching of previous responses to avoid redundant API calls.

Usage examples:

# Basic usage with async processing
python step1_prompting.py --dataset_name ise-uiuc/Magicoder-Evol-Instruct-110K --max_samples 50 --model_name gpt-4o-mini --save_batch_size 25 --max_concurrent 25

# High throughput processing
python step1_prompting.py --dataset_name ise-uiuc/Magicoder-Evol-Instruct-110K --max_samples 500 --model_name o3-mini-2025-01-31 --save_batch_size 25 --max_concurrent 15 --batch_delay 0.1

# Conservative settings for rate-limited scenarios
python step1_prompting.py --dataset_name bigcode/stack-dedup-python-fns --max_samples 100 --model_name gpt-4 --save_batch_size 5 --max_concurrent 3 --batch_delay 2.0

# Resume interrupted processing (cached items will be skipped)
python step1_prompting.py --dataset_name ise-uiuc/Magicoder-OSS-Instruct-75K --max_samples 1000 --model_name gpt-4o-mini --save_batch_size 20 --max_concurrent 10
"""