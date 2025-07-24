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
from openai import OpenAI

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
class OpenAIAsyncClient:
    """
    Async OpenAI client using aiohttp for better control and performance.
    """
    
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300),  # 5 minute timeout
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def chat_completion(
        self,
        messages: List[dict],
        model: str,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        top_p: float = 1.0,
        seed: Optional[int] = None,
    ) -> str:
        """
        Send a chat completion request to OpenAI API.
        """
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        
        if seed is not None:
            payload["seed"] = seed
        
        async with self.session.post(url, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                return data["choices"][0]["message"]["content"]
            else:
                error_text = await response.text()
                raise aiohttp.ClientResponseError(
                    request_info=response.request_info,
                    history=response.history,
                    status=response.status,
                    message=f"OpenAI API error: {error_text}"
                )


async def generate_with_retry(
    client: OpenAIAsyncClient,
    messages: List[dict],
    model: str,
    max_tokens: int,
    temperature: float = 0.7,
    top_p: float = 1.0,
    seed: Optional[int] = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    semaphore: asyncio.Semaphore = None,
) -> str:
    """
    Generate response with retry logic for handling rate limits and errors.
    """
    async def _make_request():
        for attempt in range(max_retries):
            try:
                response = await client.chat_completion(
                    messages=messages,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed
                )
                return response
            except aiohttp.ClientResponseError as e:
                if e.status == 429:  # Rate limit
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        print(f"Rate limit hit, waiting {wait_time:.1f}s before retry {attempt + 2}/{max_retries}")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        return f"ERROR: Rate limit exceeded after {max_retries} attempts"
                elif e.status >= 500:  # Server error
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        print(f"Server error {e.status}, retrying in {wait_time:.1f}s (attempt {attempt + 2}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        return f"ERROR: Server error {e.status} after {max_retries} attempts"
                else:
                    return f"ERROR: HTTP {e.status} - {str(e)}"
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"Request timeout, retrying in {wait_time:.1f}s (attempt {attempt + 2}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    return f"ERROR: Timeout after {max_retries} attempts"
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"Unexpected error: {e}, retrying in {wait_time:.1f}s (attempt {attempt + 2}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    return f"ERROR: {str(e)}"
        
        return "ERROR: All retry attempts failed"
    
    if semaphore:
        async with semaphore:
            return await _make_request()
    else:
        return await _make_request()


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
        if max_sample is not None:
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


FILE_NAME = Path(__file__).stem
default_output_dir = Path(__file__).parent / "outputs" / FILE_NAME
default_cache_dir = Path(__file__).parent / "outputs" / FILE_NAME / "cache"


async def process_batch_async(
    client: OpenAIAsyncClient,
    batch_items: List[dict],
    model_name: str,
    max_tokens: int,
    cached_data: dict,
    cache_file: Path,
    max_concurrent: int = 10,
    max_retries: int = 3,
    retry_delay: float = 1.0,
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
        hash_id = hash_messages(messages)
        
        # Check cache first
        if hash_id in cached_data:
            response = cached_data[hash_id]['response']
        else:
            # Generate new response
            response = await generate_with_retry(
                client=client,
                messages=messages,
                model=model_name,
                max_tokens=max_tokens,
                max_retries=max_retries,
                retry_delay=retry_delay,
                semaphore=semaphore
            )
            
            # Cache the result
            cache_item = {
                'hash_id': hash_id,
                'response': response
            }
            cached_data[hash_id] = cache_item
            append_jsonl(cache_file, cache_item)
        
        # Update item with response
        result_item = item.copy()
        result_item['gpt_response'] = response
        result_item['hash_id'] = hash_id
        return result_item
    
    # Process all items in the batch concurrently
    tasks = [process_single_item(item) for item in batch_items]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return results


async def main_async(
    dataset_name: str = "ise-uiuc/Magicoder-Evol-Instruct-110K",
    max_samples: Optional[int] = None,
    model_name: str = "o3-mini-2025-01-31",
    max_tokens: int = 8192,
    num_proc: int = 16,
    num_test_cases_to_generate: int = 50,
    output_dir: str = default_output_dir,
    cache_dir: str = default_cache_dir,
    overwrite: bool = False,
    batch_size: int = 20,
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
    print(f"Batch size: {batch_size}")
    print(f"Max concurrent: {max_concurrent}")
    
    # Preprocess dataset
    dataset = preprocess_dataset(dataset_name, max_sample=max_samples, num_proc=num_proc)
    data = list(dataset)  # Convert to list for easier processing
    
    # Setup paths
    cache_dir = Path(cache_dir)
    cache_file = cache_dir / f"{pretty_name(dataset_name)}_{pretty_name(model_name)}.jsonl"
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    output_file = Path(output_dir) / f"{pretty_name(dataset_name)}_{pretty_name(model_name)}.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
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
    final_results = []
    
    for item in data:
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
        
        if hash_id in cached_data:
            # Use cached result
            result_item = item.copy()
            result_item['gpt_response'] = cached_data[hash_id]['response']
            result_item['hash_id'] = hash_id
            final_results.append(result_item)
        else:
            # Needs processing
            items_to_process.append(item)
            final_results.append(item)  # Will be updated later
    
    print(f"Found {len(cached_data)} cached items, {len(items_to_process)} items need processing")
    
    if len(items_to_process) == 0:
        print("All items are cached, saving final results...")
        # Save final results
        with open(output_file, 'w') as f:
            for item in final_results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Results saved to {output_file}")
        return
    
    # Create mapping for updating final results
    items_to_process_map = {}
    final_results_indices = {}
    process_idx = 0
    
    for i, item in enumerate(data):
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
        
        if hash_id not in cached_data:
            items_to_process_map[process_idx] = i
            final_results_indices[process_idx] = i
            process_idx += 1
    
    # Process items in batches using async client
    async with OpenAIAsyncClient(api_key=api_key, base_url=base_url) as client:
        num_batches = (len(items_to_process) + batch_size - 1) // batch_size
        processed_count = 0
        
        for i in tqdm(range(0, len(items_to_process), batch_size), desc="Processing batches"):
            batch_items = items_to_process[i:i + batch_size]
            
            # Process batch
            batch_results = await process_batch_async(
                client=client,
                batch_items=batch_items,
                model_name=model_name,
                max_tokens=max_tokens,
                cached_data=cached_data,
                cache_file=cache_file,
                max_concurrent=max_concurrent,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )
            
            # Update final results
            for j, result_item in enumerate(batch_results):
                process_result_idx = i + j
                final_result_idx = final_results_indices[process_result_idx]
                final_results[final_result_idx] = result_item
            
            processed_count += len(batch_results)
            print(f"Processed batch {i//batch_size + 1}/{num_batches} ({processed_count}/{len(items_to_process)} items)")
            
            # Add delay between batches
            if batch_delay > 0 and i + batch_size < len(items_to_process):
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
    num_proc: int = 16,
    num_test_cases_to_generate: int = 50,
    output_dir: str = default_output_dir,
    cache_dir: str = default_cache_dir,
    overwrite: bool = False,
    batch_size: int = 20,
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
            num_proc=num_proc,
            num_test_cases_to_generate=num_test_cases_to_generate,
            output_dir=output_dir,
            cache_dir=cache_dir,
            overwrite=overwrite,
            batch_size=batch_size,
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
python step1_prompting.py --dataset_name ise-uiuc/Magicoder-Evol-Instruct-110K --max_samples 50 --model_name gpt-4o-mini --batch_size 10 --max_concurrent 5

# High throughput processing
python step1_prompting.py --dataset_name ise-uiuc/Magicoder-Evol-Instruct-110K --max_samples 500 --model_name o3-mini-2025-01-31 --batch_size 25 --max_concurrent 15 --batch_delay 0.1

# Conservative settings for rate-limited scenarios
python step1_prompting.py --dataset_name bigcode/stack-dedup-python-fns --max_samples 100 --model_name gpt-4 --batch_size 5 --max_concurrent 3 --batch_delay 2.0

# Resume interrupted processing (cached items will be skipped)
python step1_prompting.py --dataset_name ise-uiuc/Magicoder-OSS-Instruct-75K --max_samples 1000 --model_name gpt-4o-mini --batch_size 20 --max_concurrent 10
"""