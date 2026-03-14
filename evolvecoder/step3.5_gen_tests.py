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

from evolvecoder.utils import (
    parse_incomplete_json,
    append_jsonl,
    load_cache,
    save_gen_results,
    prepare_environment,
    load_jsonl,
    chunking,
    get_python_code_from_string,
    hash_messages,
    pretty_name,
)
from acecoderv2.synthesizer.openai_utils import generate_with_retry, OpenAIAsyncClient

PROMPT_TEMPLATE_RAW = """system:
You are an advanced AI system specialized in generating *differentiating test cases* that expose logical and behavioral differences between similar programs.
You will receive a coding problem, five programs that currently produce mostly similar evaluation results, existing test cases, and their evaluation results.

Your task is to create 20 new assert-based test cases that maximize discrimination power among these programs.

Please generate 20 new assert-based test cases that satisfy the following requirements:
- Each test case must clearly **differentiate among programs** that share the similar evaluation results.
    -> At least one program must fail in each test case.
    -> At least one program must pass in each test case.
    -> Test cases should expose different failure modes across the programs.
- All test cases must be correct according to the problem definition, not based on any specific program.
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

Evaluation results (rows = programs, columns = tests):
{eval_tests}

Output format (JSON array of strings):
{{"tests": ["assert ...", "assert ..."]}}.
"""


def select_programs(eval_lists):
    def pairwise_distance(a, b):
        return sum(x != y for x, y in zip(a, b))

    def find_identical_groups():
        n = len(eval_lists)
        groups = defaultdict(list)

        for i in range(n):
            key = tuple(eval_lists[i])
            groups[key].append(i)

        identical_groups = [indices for indices in groups.values() if len(indices) > 1]

        return identical_groups

    def select_closest_programs(n):
        if n <= 5:
            return list(range(n))

        total_distances = []
        for i in range(n):
            total_dist = sum(pairwise_distance(eval_lists[i], eval_lists[j])
                             for j in range(n) if j != i)
            total_distances.append((i, total_dist))

        sorted_by_dist = sorted(total_distances, key=lambda x: x[1])
        selected = [idx for idx, _ in sorted_by_dist[:5]]

        return selected

    def select_by_distance(selected, target_count=5):
        n = len(eval_lists)
        selected_set = set(selected)
        remaining = [i for i in range(n) if i not in selected_set]

        distance_map = defaultdict(list)
        for idx in remaining:
            min_dist = min(pairwise_distance(eval_lists[idx], eval_lists[s])
                           for s in selected)
            distance_map[min_dist].append(idx)

        for dist in sorted(distance_map.keys()):
            candidates = distance_map[dist]
            needed = target_count - len(selected)
            if needed <= 0:
                break

            if len(candidates) <= needed:
                selected.extend(candidates)
            else:
                selected.extend(random.sample(candidates, needed))

        return selected[:target_count]

    n = len(eval_lists)
    assert n >= 5, 'At least five programs per question'

    identical_groups = find_identical_groups()

    if identical_groups:
        weights = [len(group) for group in identical_groups]
        selected_group = random.choices(identical_groups, weights=weights, k=1)[0]

        if len(selected_group) >= 5:
            selected = random.sample(selected_group, 5)
            return sorted(selected)
        else:
            selected = list(selected_group)
            selected = select_by_distance(selected, target_count=5)
            return sorted(selected)
    else:
        selected = select_closest_programs(n)
        return sorted(selected)


def preprocess_dataset(file_path, max_sample=None, num_proc=4):
    dataset = datasets.Dataset.from_json(file_path)
    if max_sample:
        dataset = dataset.select(range(max_sample))
    
    def process_dataset_item(item, idx):
        eval_lists = [
            [case['pass'] for case in result['test_cases_pass_status']]
            for result in item['gen_result']['eval_results_second']
        ]

        eval_index = select_programs(eval_lists)
        assert len(eval_index) == 5, f"Expected 5, got {len(eval_index)}"

        eval_matrix = [eval_lists[i] for i in eval_index]
        eval_matrix_str = "[\n" + "\n".join(
                "    " + str(row) + "," for row in eval_matrix
            ) + "\n]"
        solutions = [item['gen_result']['eval_results_second'][i]['parse_code'] for i in eval_index]

        return {
            "id": item["id"],
            "problem": item["problem"],
            "raw_tests": item["raw_tests"],
            "outputs": item["outputs"],
            "synthesis_result_first": item['synthesis_result_first'],
            "filtered_tests_second": item['filtered_tests_second'],
            "sampled_solutions": solutions,
            "eval_matrix": eval_matrix_str,
        }
    
    dataset = dataset.map(
        process_dataset_item,
        with_indices=True,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
    )
    return dataset


async def main_async(
    file_path: str,
    max_samples: Optional[int] = None,
    model_name: str = "o3-mini-2025-01-31",
    round: int = 0,
    max_tokens: int = 8192,
    num_proc: int = 4,
    output_dir: str = Path(__file__).parent / "outputs",
    overwrite: bool = False,
    save_batch_size: int = 20,
    max_concurrent: int = 10,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    batch_delay: float = 0.5,
    api_key: Optional[str] = None,
    base_url: str = "https://api.openai.com/v1",
    **kwargs
    ):
    
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
    
    FILE_NAME = Path(__file__).stem
    cache_file, output_file = prepare_environment(
        FILE_NAME, file_path, output_dir, round, overwrite
    )
    if cache_file is None and output_file is None:
        return
    data = preprocess_dataset(file_path, max_samples, num_proc)
    cached_data = load_cache(cache_file)
    
    def get_items_to_gen(data, cached_data):
        items_to_process = []
        items_to_process_map = {}
        final_results = []

        for i, item in enumerate(data):
            filtered_tests_str = "[\n" + "\n".join(
                    "    " + str(row) + "," for row in item['filtered_tests_second']
                ) + "\n]"
            prompt = PROMPT_TEMPLATE_RAW.format(
                question=item['problem'],
                tests=filtered_tests_str,
                program1=item['sampled_solutions'][0],
                program2=item['sampled_solutions'][1],
                program3=item['sampled_solutions'][2],
                program4=item['sampled_solutions'][3],
                program5=item['sampled_solutions'][4],
                eval_tests=item['eval_matrix']
            )
            messages = [{"role": "user", "content": prompt}]
            hash_id = hash_messages(messages)
            item['synthesis_result_second'] = {"hash_id": hash_id}

            if hash_id in cached_data:
                item['synthesis_result_second']['gpt_response'] = cached_data[hash_id]['gpt_response']
                item['synthesis_result_second']['gpt_prompt'] = cached_data[hash_id]['gpt_prompt']
            else:
                items_to_process_map[hash_id] = i
                items_to_process.append(item)

            final_results.append(item)

        return items_to_process, items_to_process_map, final_results

    items_to_process, items_to_process_map, final_results = get_items_to_gen(data, cached_data)

    async with OpenAIAsyncClient(api_key=api_key, base_url=base_url) as client:
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_item(item):
            filtered_tests_str = "[\n" + "\n".join(
                    "    " + str(row) + "," for row in item['filtered_tests_second']
                ) + "\n]"
            prompt = PROMPT_TEMPLATE_RAW.format(
                question=item['problem'],
                tests=filtered_tests_str,
                program1=item['sampled_solutions'][0],
                program2=item['sampled_solutions'][1],
                program3=item['sampled_solutions'][2],
                program4=item['sampled_solutions'][3],
                program5=item['sampled_solutions'][4],
                eval_tests=item['eval_matrix']
            )
            messages = [{"role": "user", "content": prompt}]
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
            item['synthesis_result_second']['gpt_prompt'] = prompt
            item['synthesis_result_second']['gpt_response'] = response[0]
            return item

        for i in tqdm(range(0, len(items_to_process), save_batch_size), desc="Processing batches"):
            batch = items_to_process[i:i + save_batch_size]
            tasks = [process_single_item(item) for item in batch]
            batch_results = await tqdm.gather(*tasks, desc="Processing batch", total=len(tasks))
            
            append_jsonl(cache_file, [i['synthesis_result_second'] for i in batch_results])
            for result in batch_results:
                idx = items_to_process_map[result['synthesis_result_second']['hash_id']]
                final_results[idx] = result

            if batch_delay > 0:
                await asyncio.sleep(batch_delay)

    save_gen_results(final_results, output_file, cache_file)


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
python evolvecoder/step3.6_gen_tests.py evolvecoder/outputs/all_20_round1/gpt_4.1_mini/step3.5_filter_tests_round_1.jsonl --round 1 --max_samples 2 --model_name gpt-4.1-mini --save_batch_size 1 --max_concurrent 1 --batch_delay 2.0
"""