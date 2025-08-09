# import fire
# import json
# import os
# import asyncio
# import aiohttp
# from pathlib import Path
# from typing import Optional, List, Dict, Any
# from tqdm.asyncio import tqdm
# import time
# import sys
# from pathlib import Path

# from utils import pretty_name, append_jsonl, hash_messages
# from openai_utils import generate_with_retry_sync, OpenAISyncClient 



# def load_previous_tests(file_path: str) -> Dict[str, List[str]]:
#     previous_data = load_jsonl(file_path)
#     hash_to_tests = {}
#     for item in previous_data:
#         hash_id = item.get("gen_result", {}).get("qid")
#         outputs = item.get("gen_result", {}).get("outputs", [])
#         if hash_id and outputs:
#             hash_to_tests[hash_id] = outputs
#     return hash_to_tests



# def preprocess_prompts_auto(data: List[dict], prev_tests: Dict[str, List[str]]) -> tuple[List[List[dict]], List[str]]:
#     """
#     Preprocess prompts for OpenAI API format.
#     """
#     messages_list = []
#     qid_list = []
#     for item in data:
#         qid = item['gen_result']['qid']
#         prev_cases = prev_tests.get(qid, [])
#         if prev_cases:
#             prompt = (
#                 f"The following is a challenging coding problem:\n\n"
#                 f"{item['synthesis_result']['problem']}\n\n"
#                 f"The following test cases were previously generated:\n"
#                 + "\n".join(f"- {c}" for c in prev_cases)
#                 + "\n\nPlease generate more **challenging and edge-case** test cases that go beyond the existing ones."
#             )
#         else:
#             prompt = f"Generate challenging test cases for the following problem:\n\n{item['synthesis_result']['problem']}"
        
#         messages = [{"role": "user", "content": item['synthesis_result']["problem"]}]
#         messages_list.append(messages)
#         qid_list.append(qid)
#     return messages_list, qid_list


# def preprocess_prompts(data: List[dict], mode: str = "auto", prev_tests: Dict[str, List[str]] = {}) -> tuple[List[List[dict]], List[str]]:
#     if mode == "auto":
#         return preprocess_prompts_auto(data, prev_tests)
#     else:
#         raise ValueError(f"Unsupported mode: {mode}. Supported modes: 'auto'.")
    

# async def process_batch(
#     client: OpenAISyncClient,
#     messages_batch: List[List[dict]],
#     qids_batch: List[str],
#     model: str,
#     temperature: float,
#     top_p: float,
#     n: int,
#     max_tokens: int,
#     seed: int,
#     max_retries: int,
#     retry_delay: float,
#     max_concurrent: int,
# ) -> List[str]:
#     """
#     Process a batch of requests concurrently with controlled concurrency.
#     """
#     semaphore = asyncio.Semaphore(max_concurrent)
    
#     tasks = []
#     for messages in messages_batch:
#         task = generate_with_retry_sync(
#             client=client,
#             messages=messages,
#             model=model,
#             temperature=temperature,
#             top_p=top_p,
#             n=n,
#             max_tokens=max_tokens,
#             seed=seed,
#             max_retries=max_retries,
#             retry_delay=retry_delay,
#             semaphore=semaphore
#         )
#         tasks.append(task)
    
#     # Wait for all tasks to complete
#     responses = await tqdm.gather(*tasks, desc="Processing batch", unit="request", total=len(tasks))
#     return responses


# # FILE_NAME = Path(__file__).stem
# FILE_NAME = "step2.1_gen"
# default_output_dir = Path(__file__).parent / "outputs" / FILE_NAME
# default_cache_dir = Path(__file__).parent / "outputs" / FILE_NAME / "cache"


# async def main_async(
#     file_path: str,
#     output_dir: str = None,
#     cache_dir: str = None,
#     start_idx: int = 0,
#     end_idx: Optional[int] = None,
#     batch_size: int = 20,  # Increased default batch size
#     max_concurrent: int = 10,  # Increased default concurrency
#     model: str = "gpt-3.5-turbo",
#     api_key: Optional[str] = None,
#     base_url: str = "https://api.openai.com/v1",
#     seed: int = 42,
#     top_p: float = 0.95,
#     n: int = 1,
#     temperature: float = 0.6,
#     max_tokens: int = 4000,
#     overwrite: bool = False,
#     max_retries: int = 3,
#     retry_delay: float = 1.0,
#     batch_delay: float = 0.5,
#     progress_bar: bool = True,
#     prev_testcase_file: Optional[str] = None
# ):
#     if api_key is None:
#         api_key = os.getenv("OPENAI_API_KEY")
#         if api_key is None:
#             raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
    
#     prev_tests = {}
#     if prev_testcase_file:
#         from acecoderv2.synthesizer.utils import load_jsonl
#         prev_tests = load_previous_tests(prev_testcase_file)
#         print(f"Loaded previous test cases for {len(prev_tests)} items")

#     output_dir = Path(output_dir) if output_dir else Path(file_path).parent
#     output_dir.mkdir(parents=True, exist_ok=True)

#     if start_idx is None and end_idx is None:
#         cache_file = output_dir / f"{FILE_NAME}_{pretty_name(model)}_seed{seed}.cache.jsonl"
#         output_file = output_dir / f"{FILE_NAME}_{pretty_name(model)}_seed{seed}.jsonl"
#     else:
#         if end_idx is None:
#             end_idx = len(data)
#         cache_file = output_dir / f"{FILE_NAME}_{pretty_name(model)}_seed{seed}_{start_idx}_{end_idx}.cache.jsonl"
#         output_file = output_dir / f"{FILE_NAME}_{pretty_name(model)}_seed{seed}_{start_idx}_{end_idx}.jsonl"

#     if output_file.exists() and output_file.stat().st_size != 0 and not overwrite:
#         print(f"Output file {output_file} already exists. Use --overwrite to overwrite.")
#         return
    
#     # Load cached data if exists
#     cached_data = {}
#     if cache_file.exists() and not overwrite:
#         with open(cache_file, 'r') as f:
#             for line in f.readlines():
#                 if line.strip():
#                     item = json.loads(line)
#                     cached_data[item['qid']] = item
#         print(f"Loaded {len(cached_data)} cached items from {cache_file}")
    
#     # Load data
#     if file_path.endswith('.jsonl'):
#         with open(file_path, 'r') as f:
#             data = [json.loads(line) for line in f.readlines()]
#     elif file_path.endswith('.json'):
#         with open(file_path, 'r') as f:
#             data = json.load(f)
#     else:
#         raise ValueError("Unsupported file format. Please provide a .jsonl or .json file.")

#     if start_idx is not None and end_idx is not None:
#         data = data[start_idx:end_idx]
    
#     # Identify items that need processing (not in cache)
#     items_to_process = []
#     final_results = []
    
#     for item in data:
#         qid = hash_messages(item['synthesis_result']['problem'])
#         if qid in cached_data:
#             # Use cached result
#             new_item = item.copy()
#             new_item['gen_result'] = cached_data[qid]
#         else:
#             new_item = item.copy()
#             new_item['gen_result'] = {
#                 'outputs': [],
#                 'qid': qid,
#                 'prompt': None,
#                 'sampling_params': {
#                     "model_name_or_path": model,
#                     "temperature": temperature,
#                     "top_p": top_p,
#                     "n": n,
#                     "max_tokens": max_tokens,
#                     "seed": seed,
#                 }
#             }
#             # Needs processing
#             items_to_process.append(new_item)
#         final_results.append(new_item)  # Will be updated with results later
    
#     print(f"Processing {len(data)} items from {start_idx} to {end_idx}...")
#     print(f"Found {len(cached_data)} cached items, {len(items_to_process)} items need processing")
    
#     if len(items_to_process) == 0:
#         print("All items are cached, saving final results...")
#         # Save final results
#         with open(output_file, 'w') as f:
#             for item in final_results:
#                 f.write(json.dumps(item, ensure_ascii=False) + '\n')
#         print(f"Results saved to {output_file}")
#         return
    
#     print(f"Model: {model}")
#     print(f"Base URL: {base_url}")
#     print(f"Temperature: {temperature}")
#     print(f"Top-p: {top_p}")
#     print(f"Max tokens: {max_tokens}")
#     print(f"Seed: {seed}")
#     print(f"Batch size: {batch_size}")
#     print(f"Max concurrent requests per batch: {max_concurrent}")

#     # Preprocess prompts for items that need processing
#     messages_list, qids = preprocess_prompts(items_to_process, prev_tests=prev_tests)

#     qids = [item['gen_result']['qid'] for item in items_to_process]
    
#     # Create mapping for updating final results
#     qid_to_result_idx = {item['gen_result']['qid']: idx for idx, item in enumerate(final_results)}

#     # Create async client context
#     start_time = time.time()
#     async with OpenAISyncClient(api_key=api_key, base_url=base_url) as client:
#         # Process in batches
#         total_processed = 0
#         num_batches = (len(messages_list) + batch_size - 1) // batch_size
        
#         batch_iterator = range(0, len(messages_list), batch_size)
#         if progress_bar:
#             batch_iterator = tqdm(batch_iterator, desc="Processing batches", unit="batch")
        
#         for i in batch_iterator:
#             batch_messages = messages_list[i:i + batch_size]
#             batch_qids = qids[i:i + batch_size]
#             batch_data = items_to_process[i:i + batch_size]
            
#             if not progress_bar:
#                 print(f"\nProcessing batch {i//batch_size + 1}/{num_batches}")
            
#             # Generate responses for this batch
#             batch_start_time = time.time()
#             batch_responses = await process_batch(
#                 client=client,
#                 messages_batch=batch_messages,
#                 qids_batch=batch_qids,
#                 model=model,
#                 temperature=temperature,
#                 top_p=top_p,
#                 n=n,
#                 max_tokens=max_tokens,
#                 seed=seed,
#                 max_retries=max_retries,
#                 retry_delay=retry_delay,
#                 max_concurrent=max_concurrent
#             )
#             batch_time = time.time() - batch_start_time
            
#             # Process responses and update data
#             batch_results = []
#             for j, responses in enumerate(batch_responses):
#                 # Update the item with results
#                 batch_data[j]['gen_result']['outputs'].extend(responses)
#                 batch_data[j]['gen_result']['prompt'] = batch_messages[j]

#                 # Update final results
#                 result_idx = qid_to_result_idx[batch_qids[j]]
#                 final_results[result_idx] = batch_data[j]

#                 batch_results.append(batch_data[j]['gen_result'])

#             # Save batch to cache
#             append_jsonl(cache_file, batch_results)
            
#             total_processed += len(batch_messages)
#             if not progress_bar:
#                 print(f"Completed {total_processed}/{len(messages_list)} items in {batch_time:.1f}s")
#                 print(f"Throughput: {len(batch_messages)/batch_time:.1f} requests/second")
#                 print(f"Saved batch {i//batch_size + 1} to cache ({len(batch_results)} items)")
            
#             # Add delay between batches to be respectful to the API
#             if batch_delay > 0 and i + batch_size < len(messages_list):
#                 await asyncio.sleep(batch_delay)

#     # Save final results
#     with open(output_file, 'w') as f:
#         for item in final_results:
#             f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
#     total_time = time.time() - start_time
#     print(f"\nGenerated responses saved to {output_file}")
#     print(f"Total processing time: {total_time:.1f}s")
#     if len(items_to_process) > 0:
#         print(f"Average throughput: {len(items_to_process)/total_time:.1f} requests/second")
    
#     # Remove cache file
#     if cache_file.exists():
#         os.remove(cache_file)
#         print(f"Cache file {cache_file} removed")


# def main(
#     file_path: str,
#     output_dir: str = None,
#     cache_dir: str = None,
#     start_idx: int = None,
#     end_idx: Optional[int] = None,
#     batch_size: int = 20,
#     max_concurrent: int = 10,
#     model: str = "gpt-3.5-turbo",
#     api_key: Optional[str] = None,
#     base_url: str = "https://api.openai.com/v1",
#     seed: int = 42,
#     top_p: float = 0.95,
#     n: int = 1,
#     temperature: float = 0.6,
#     max_tokens: int = 4000,
#     overwrite: bool = False,
#     max_retries: int = 3,
#     retry_delay: float = 1.0,
#     batch_delay: float = 0.5,
#     progress_bar: bool = True,
#     prev_testcase_file: Optional[str] = None
# ):
#     """
#     Synchronous wrapper for the async main function.
#     """
#     start_time = time.time()
    
#     try:
#         asyncio.run(main_async(
#             file_path=file_path,
#             output_dir=output_dir,
#             cache_dir=cache_dir,
#             start_idx=start_idx,
#             end_idx=end_idx,
#             batch_size=batch_size,
#             max_concurrent=max_concurrent,
#             model=model,
#             api_key=api_key,
#             base_url=base_url,
#             seed=seed,
#             top_p=top_p,
#             n=n,
#             temperature=temperature,
#             max_tokens=max_tokens,
#             overwrite=overwrite,
#             max_retries=max_retries,
#             retry_delay=retry_delay,
#             batch_delay=batch_delay,
#             progress_bar=progress_bar,
#             prev_testcase_file=prev_testcase_file
#         ))
#     except KeyboardInterrupt:
#         print("\nOperation cancelled by user")
#     except Exception as e:
#         print(f"\nError: {e}")
#         raise


# if __name__ == "__main__":
#     fire.Fire(main)

# """
# Usage examples:

# # High performance settings with aiohttp
# python step2.1_openai_gen.py outputs/Magicoder_Evol_Instruct_110K/gpt_4o_mini/step1.1_parsing.jsonl \
#     --start_idx=0 \
#     --end_idx=50 \
#     --batch_size=10 \
#     --max_concurrent=25 \
#     --model='gpt-4.1-mini' \
#     --top_p=0.95 \
#     --temperature=0.6 \
#     --max_tokens=4000 \
#     --n=4

# # Conservative settings for rate limit sensitive scenarios
# python step2.1_openai_gen.py outputs/Magicoder_Evol_Instruct_110K/gpt_4o_mini/step1.1_parsing.jsonl \
#     --start_idx=0 \
#     --end_idx=100 \
#     --batch_size=10 \
#     --max_concurrent=5 \
#     --model='gpt-4' \
#     --batch_delay=2.0

# # Using custom OpenAI-compatible API
# python step2.1_openai_gen.py outputs/Magicoder_Evol_Instruct_110K/gpt_4o_mini/step1.1_parsing.jsonl \
#     --start_idx=0 \
#     --end_idx=500 \
#     --batch_size=30 \
#     --max_concurrent=15 \
#     --model='llama-3-70b' \
#     --base_url='https://your-custom-endpoint.com/v1' \
#     --api_key='your-api-key'

# # Maximum throughput (be careful with rate limits!)
# python step2.1_openai_gen.py outputs/Magicoder_Evol_Instruct_110K/gpt_4o_mini/step1.1_parsing.jsonl \
#     --start_idx=0 \
#     --end_idx=2000 \
#     --batch_size=100 \
#     --max_concurrent=50 \
#     --batch_delay=0.1 \
#     --retry_delay=0.5

# # Resume interrupted job (cached items will be skipped)
# python step2.1_openai_gen.py outputs/Magicoder_Evol_Instruct_110K/gpt_4o_mini/step1.1_parsing.jsonl \
#     --start_idx=0 \
#     --end_idx=1000 \
#     --batch_size=25 \
#     --max_concurrent=15 \
#     --model='gpt-4o-mini'
# """

import fire
import json
import os
from pathlib import Path
from typing import Optional, List, Dict
from tqdm import tqdm
import time

from utils import pretty_name, append_jsonl, hash_messages, load_jsonl
from openai_utils import generate_with_retry_sync, OpenAISyncClient

def load_previous_tests(file_path: str) -> Dict[str, List[str]]:
    previous_data = load_jsonl(file_path)
    hash_to_tests = {}
    for item in previous_data:
        hash_id = item.get("gen_result", {}).get("qid")
        outputs = item.get("gen_result", {}).get("outputs", [])
        if hash_id and outputs:
            hash_to_tests[hash_id] = outputs
    return hash_to_tests

def preprocess_prompts(data: List[dict], prev_tests: Dict[str, List[str]], generation_mode: str = "programs") -> tuple[List[List[dict]], List[str]]:
    messages_list = []
    qid_list = []
    for item in data:
        qid = item['gen_result']['qid']
        
        # Safe extraction of problem
        problem = item.get('synthesis_result', {}).get('problem') or item.get('problem', '')
        
        if generation_mode == "test_cases":
            # Generate test cases (even rounds: 2,4,6...)
            prompt = f"Generate test cases for:\n\n{problem}"
        elif generation_mode == "programs":
            # Generate programs (odd rounds: 1,3,5...)
            tests = item.get('synthesis_result', {}).get('tests', [])
            if tests:
                tests_str = "\n".join(f"- {test}" for test in tests)
                prompt = f"""Given the following problem and test cases, generate alternative Python solutions that can pass these tests:

Problem:
{problem}

Test Cases:
{tests_str}

Please generate multiple different Python solutions (variations in approach, algorithm, or implementation style) that solve this problem and pass all the given test cases."""
            else:
                prompt = f"Generate Python solutions for:\n\n{problem}"
        else:
            raise ValueError(f"Unknown generation_mode: {generation_mode}")
        
        messages = [{"role": "user", "content": prompt}]
        messages_list.append(messages)
        qid_list.append(qid)
    return messages_list, qid_list

def main(
    file_path: str,
    output_dir: str = None,
    model: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None,
    base_url: str = "https://api.openai.com/v1",
    seed: int = 42,
    top_p: float = 0.95,
    n: int = 5,
    temperature: float = 0.6,
    max_tokens: int = 4000,
    overwrite: bool = False,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    prev_testcase_file: Optional[str] = None,
    generation_mode: str = "programs"  # New parameter: "test_cases" or "programs" (default to programs for backward compatibility)
):
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OpenAI API key not found.")

    prev_tests = {}
    if prev_testcase_file:
        prev_tests = load_previous_tests(prev_testcase_file)
        print(f"Loaded previous test cases for {len(prev_tests)} items")

    output_dir = Path(output_dir) if output_dir else Path(file_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"step2.1_gen_{pretty_name(model)}_seed{seed}.jsonl"

    if output_file.exists() and output_file.stat().st_size != 0 and not overwrite:
        print(f"Output file {output_file} already exists. Use --overwrite to overwrite.")
        return

    # Load data
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]

    items_to_process = []
    final_results = []

    for item in data:
        # Safe extraction of problem with fallback
        problem = item.get('synthesis_result', {}).get('problem') or item.get('problem', '')
        if not problem:
            print(f"Warning: Skipping item with missing problem: {item.get('id', 'unknown')}")
            continue
            
        qid = hash_messages(problem)
        item['gen_result'] = {
            'outputs': [],
            'qid': qid,
            'prompt': None,
            'sampling_params': {
                "model_name_or_path": model,
                "temperature": temperature,
                "top_p": top_p,
                "n": n,
                "max_tokens": max_tokens,
                "seed": seed,
            }
        }
        items_to_process.append(item)
        final_results.append(item)

    print(f"Processing {len(items_to_process)} items...")
    client = OpenAISyncClient(api_key=api_key, base_url=base_url)
    messages_list, qids = preprocess_prompts(items_to_process, prev_tests, generation_mode)

    desc = f"Generating {generation_mode}"
    for i, item in enumerate(tqdm(items_to_process, desc=desc)):
        responses = generate_with_retry_sync(
            openai_client=client.client,
            messages=messages_list[i],
            model=model,
            temperature=temperature,
            top_p=top_p,
            n=n,
            max_tokens=max_tokens,
            seed=seed
        )

        if generation_mode == "programs":
            # For program generation, store the generated programs
            item['gen_result']['generated_programs'] = responses
            item['gen_result']['outputs'].extend(responses)  # Also keep in outputs for backward compatibility
        else:
            # For test case generation mode:
            # 1. Keep existing programs (don't replace them)
            # 2. Generate new test cases and add them to synthesis_result.tests
            # 3. Store new test case responses in outputs for record keeping
            
            # Only extend outputs if they don't contain programs already
            if 'eval_results' not in item['gen_result'] or not item['gen_result']['eval_results']:
                item['gen_result']['outputs'].extend(responses)
            else:
                # We already have programs, so we're generating test cases
                # Store test case generation responses separately to avoid confusion
                if 'test_case_outputs' not in item['gen_result']:
                    item['gen_result']['test_case_outputs'] = []
                item['gen_result']['test_case_outputs'].extend(responses)
            
            # 🔧 FIX: Parse test cases and add them to synthesis_result.tests
            new_test_cases = []
            for response in responses:
                if isinstance(response, dict):
                    # Handle different response formats
                    if 'message' in response and 'content' in response['message']:
                        content = response['message']['content']
                    elif 'content' in response:
                        content = response['content']
                    else:
                        continue
                    
                    # Extract test cases from the response content
                    import re
                    
                    # Look for various test case patterns
                    patterns = [
                        r'assert\s+[^\n;]+',  # assert statements
                        r'def\s+test_[^:]+:[^}]+',  # test functions  
                        r'n\s*=\s*\d+[^}]*(?:edges|weights)[^}]*',  # input format descriptions
                        r'Input:\s*```[^`]*```',  # input code blocks
                        r'Expected Output:\s*```[^`]*```',  # expected output blocks
                    ]
                    
                    for pattern in patterns:
                        matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
                        new_test_cases.extend(matches)
                    
                    # If no structured patterns found, try to extract from the content directly
                    if not new_test_cases:
                        # Convert descriptive test cases to assert format
                        # Look for test case sections
                        test_case_sections = re.findall(r'### Test Case \d+[^#]*', content, re.MULTILINE | re.DOTALL)
                        for section in test_case_sections:
                            # Try to extract input and expected output
                            input_match = re.search(r'n\s*=\s*(\d+)', section)
                            edges_match = re.search(r'edges\s*=\s*(\[[^\]]+\])', section)
                            weights_match = re.search(r'weights\s*=\s*(\[[^\]]+\])', section)
                            
                            if input_match and edges_match and weights_match:
                                n = input_match.group(1)
                                edges = edges_match.group(1)
                                weights = weights_match.group(1)
                                test_case = f"# Test: n={n}, edges={edges}, weights={weights}"
                                new_test_cases.append(test_case)
            
            # Add new test cases to synthesis_result.tests
            if new_test_cases and 'synthesis_result' in item:
                current_tests = item['synthesis_result'].get('tests', [])
                # Remove duplicates and add new ones
                existing_test_set = set(current_tests)
                unique_new_tests = [test for test in new_test_cases if test not in existing_test_set]
                item['synthesis_result']['tests'] = current_tests + unique_new_tests
                print(f"🧪 Added {len(unique_new_tests)} new test cases for {item['gen_result']['qid']} (total: {len(item['synthesis_result']['tests'])})")
        
        item['gen_result']['prompt'] = messages_list[i]

    # Save results
    with open(output_file, 'w') as f:
        for item in final_results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\u2705 Results saved to {output_file}")

if __name__ == "__main__":
    fire.Fire(main)
