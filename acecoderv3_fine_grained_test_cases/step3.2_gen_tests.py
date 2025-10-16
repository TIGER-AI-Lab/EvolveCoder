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
You are an advanced AI system specialized in generating *adversarial and diverse* test cases that reveal subtle weaknesses in high-performing programs.
You will receive a coding problem, five different programs solving it, existing test cases, and their evaluation results.

Your task is to create 20 new assert-based test cases that significantly upgrade the current test suite.

The new test cases must satisfy the following requirements:
- Focus on **challenging high-pass-rate programs** — design cases likely to make at least one top-performing solution fail.
- Include **diverse input patterns** that challenge different dimensions of the problem space.
- **Avoid repetition** — do not duplicate existing test cases or trivial variations.
- All test cases must be correct according to the problem definition, not based on any specific program.
- Be independent of other test cases.
- Use constant values (no randomness or external resource calls).
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

    def compute_pass_rates():
        return {i: sum(lst)/len(lst) if lst else 0 for i, lst in enumerate(eval_lists)}

    def pairwise_distance(a, b):
        return sum(x != y for x, y in zip(a, b))

    def select_diverse_subset(candidates, k=3):
        if len(candidates) <= k:
            return candidates
        best, max_min, max_total = None, -1, -1
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                for l in range(j + 1, len(candidates)):
                    a, b, c = candidates[i], candidates[j], candidates[l]
                    d_ab = pairwise_distance(eval_lists[a], eval_lists[b])
                    d_ac = pairwise_distance(eval_lists[a], eval_lists[c])
                    d_bc = pairwise_distance(eval_lists[b], eval_lists[c])
                    min_d, total_d = min(d_ab, d_ac, d_bc), d_ab + d_ac + d_bc
                    if (min_d > max_min) or (min_d == max_min and total_d > max_total):
                        best, max_min, max_total = (a, b, c), min_d, total_d
        return list(best)

    assert len(eval_lists) >= 5, 'At least five programs per quesrion'

    pass_rates = compute_pass_rates()
    sorted_by_rate = sorted(pass_rates.items(), key=lambda x: x[1], reverse=True)
    top_two = [idx for idx, _ in sorted_by_rate[:2]]

    candidates = [i for i in pass_rates if i not in top_two and 0.4 <= pass_rates[i] < 0.9]
    if len(candidates) >= 3:
        selected = select_diverse_subset(candidates)
    else:
        extra = [i for i in pass_rates if i not in top_two and pass_rates[i] > 0.1]
        selected = random.sample(extra, min(3, len(extra)))
    return sorted(top_two + selected)


def preprocess_dataset(file_path, max_sample=None, num_proc=4):
    dataset = datasets.Dataset.from_json(file_path)
    if max_sample:
        dataset = dataset.select(range(max_sample))
    
    def process_dataset_item(item, idx):
        tests = item['filtered_tests']

        eval_lists = [
            [case['pass'] for case in result['test_cases_pass_status']]
            for result in item['gen_result']['eval_results']
        ]

        eval_index = select_programs(eval_lists)
        assert len(eval_index) == 5, f"Expected 5, got {len(eval_index)}"

        eval_matrix = [eval_lists[i] for i in eval_index]
        eval_matrix_str = "[\n" + "\n".join(
                "    " + str(row) + "," for row in eval_matrix
            ) + "\n]"
        solutions = [item['gen_result']['eval_results'][i]['parse_code'] for i in eval_index]

        return {
            "id": item["id"],
            "problem": item["problem"],
            "raw_tests": item["raw_tests"],
            "outputs": item["outputs"],
            "filtered_tests": tests,
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
    data = preprocess_dataset(file_path, max_samples, num_proc)
    cached_data = load_cache(cache_file)
    
    def get_items_to_gen(data, cached_data):
        items_to_process = []
        items_to_process_map = {}
        final_results = []

        for i, item in enumerate(data):
            filtered_tests_str = "[\n" + "\n".join(
                    "    " + str(row) + "," for row in item['filtered_tests']
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
            item['synthesis_result'] = {"hash_id": hash_id}

            if hash_id in cached_data:
                item['synthesis_result']['gpt_response'] = cached_data[hash_id]['gpt_response']
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
                    "    " + str(row) + "," for row in item['filtered_tests']
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
            item['synthesis_result']['gpt_prompt'] = prompt
            item['synthesis_result']['gpt_response'] = response[0]
            return item

        for i in tqdm(range(0, len(items_to_process), save_batch_size), desc="Processing batches"):
            batch = items_to_process[i:i + save_batch_size]
            tasks = [process_single_item(item) for item in batch]
            batch_results = await tqdm.gather(*tasks, desc="Processing batch", total=len(tasks))
            
            append_jsonl(cache_file, [i['synthesis_result'] for i in batch_results])
            for result in batch_results:
                idx = items_to_process_map[result['synthesis_result']['hash_id']]
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
python acecoderv3_fine_grained_test_cases/step3.2_gen_tests.py acecoderv3_fine_grained_test_cases/outputs/all_20_round1/gpt_4.1_mini/step3.1_filter_tests_round_1.jsonl --round 1 --max_samples 2 --model_name gpt-4.1-mini --save_batch_size 1 --max_concurrent 1 --batch_delay 2.0
"""