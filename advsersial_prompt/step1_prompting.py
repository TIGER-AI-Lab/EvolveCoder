import os
import requests
import json
from fire import Fire
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional, Dict
import sys
from datasets import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from synthesizer.utils import (
    parse_incomplete_json,
    append_jsonl,
    load_jsonl,
    chunking,
    get_python_code_from_string,
    hash_messages,
    pretty_name
)
from openai_utils import generate_with_retry_sync, OpenAISyncClient  # <- 你需要使用我们前面写好的同步版本

# api_key = "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"
# base_url = "https://api.openai.com/v1"
def load_previous_tests(file_path: str) -> Dict[str, List[str]]:
    previous_data = load_jsonl(file_path)
    hash_to_tests = {}
    for item in previous_data:
        try:
            hash_id = item["synthesis_result"]["hash_id"]
            tests = item["synthesis_result"]["gpt_response"]["tests"]
            if isinstance(hash_id, str) and isinstance(tests, list) and tests:
                hash_to_tests[hash_id] = tests
        except (KeyError, TypeError):
            continue
    return hash_to_tests




PROMPT_TEMPLATE_RAW_COPY = """system:
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

PROMPT_TEMPLATE_GENERATE_PROGRAMS = """system:
You are an expert programmer tasked with generating alternative Python solutions for challenging programming problems. Given a problem description and existing test cases that current programs are failing, create new and diverse solution approaches.

1. Generate 3-5 alternative Python solutions that:
   - Use different algorithmic approaches (e.g., iterative vs recursive, different data structures)
   - Have different implementation styles but solve the same problem
   - Are designed to pass the given test cases
   - Are complete, executable functions
   - Follow Python best practices

2. Each solution should:
   - Be self-contained (no external imports unless necessary)
   - Include proper function definitions
   - Handle edge cases appropriately
   - Use clear variable names and logic

user:
Problem:
{problem}

Existing test cases that need to be satisfied:
{test_cases}

Generate alternative solutions in the following JSON format:
{{"programs": ["def solution1(...):\\n    # Implementation 1", "def solution2(...):\\n    # Implementation 2", ...]}}
"""

PROMPT_TEMPLATE_RAW = """system:
You are the latest and best bot aimed at transforming some code snippet into a very challenging LeetCode-style question intended for advanced CS university students and experienced software engineers. You will be provided with a prompt for writing code, previous test cases, along with a reference program that attempts to answer the question. Please complete the following for me:
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

Here is a set of previously generated test cases:
```python
{test_cases}
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

# def preprocess_dataset(dataset_name: str, max_sample=5, num_proc=4):
#     dataset_path = '/Users/heyangfan/Desktop/AceCoderV2-main/acecoderv2/advsersial_prompt/data-evol_instruct-decontaminated.jsonl'
#     print(f"📄 Loading dataset from local JSONL file: {dataset_path}")
#     data = load_jsonl(dataset_path)
#     if max_sample is not None:
#         data = data[:max_sample]
#     def process_item(item, idx):
#         return {
#             "id": idx,
#             "problem": item.get("instruction") or item.get("problem"),
#             "response": item.get("response") or item.get("solution"),
#             "program": get_python_code_from_string(item.get("response") or item.get("solution", ""))
#         }
#     processed = [process_item(item, idx) for idx, item in enumerate(data)]
#     return processed

def preprocess_dataset(dataset_name: str, max_sample=5, num_proc=1):
    print(f"📦 Loading dataset from Hugging Face: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")

    # Convert HuggingFace dataset to list of dicts
    data = dataset.select(range(max_sample)) if max_sample is not None else dataset

    try:
        # Convert to Python list
        data = list(data)
    except Exception:
        # Fallback: iterate and cast if direct conversion fails
        data = [dict(item) for item in data]

    if not data:
        raise ValueError("Loaded dataset is empty.")

    if isinstance(data[0], str):
        data = [json.loads(row) for row in data]

    def process_item(item, idx):
        return {
            "id": idx,
            "problem": item.get("instruction") or item.get("problem"),
            "response": item.get("response") or item.get("solution"),
            "program": get_python_code_from_string(item.get("response") or item.get("solution", ""))
        }

    return [process_item(item, idx) for idx, item in enumerate(data)]

def process_batch_sync(
    client: OpenAISyncClient,
    batch_items: List[dict],
    model_name: str,
    max_tokens: int,
    cache_file: Path,
    previous_tests_dict: Dict[str, List[str]],
    generation_mode: str = "questions_and_tests",  # Add generation_mode parameter
    max_retries: int = 3,
    retry_delay: float = 1.0,
    temperature: float = 0.6,
    top_p: float = 0.95,
    seed: int = 42
) -> List[dict]:
    results = []
    for item in tqdm(batch_items, desc="Processing batch items"):
        hash_id = item['synthesis_result']['hash_id']
        prev_tests = previous_tests_dict.get(hash_id, [])
        prev_tests_str = "\n".join(prev_tests) if isinstance(prev_tests, list) else ""

        if generation_mode == "questions_and_tests":
            # Original behavior: generate questions and test cases
            if item['problem'] is not None:
                prompt = PROMPT_TEMPLATE_RAW.format(
                    program=item['program'],
                    instruction=item['problem'],
                    test_cases=prev_tests_str
                )
            else:
                prompt = PROMPT_TEMPLATE_NO_INSTRUCTION.format(
                    program=item['program']
                )
        elif generation_mode == "programs":
            # Generate alternative programs given problem and test cases
            problem = item.get('synthesis_result', {}).get('problem') or item.get('problem', '')
            test_cases = item.get('synthesis_result', {}).get('tests', [])
            test_cases_str = "\n".join(test_cases) if test_cases else ""
            
            prompt = PROMPT_TEMPLATE_GENERATE_PROGRAMS.format(
                problem=problem,
                test_cases=test_cases_str
            )
        elif generation_mode == "test_cases":
            # Generate test cases (similar to step2.1 logic)
            problem = item.get('synthesis_result', {}).get('problem') or item.get('problem', '')
            prompt = f"Generate challenging test cases for the following problem:\n\n{problem}"
        else:
            raise ValueError(f"Unknown generation_mode: {generation_mode}")

        messages = [{"role": "user", "content": prompt}]
        # response = generate_with_retry_sync(
        #     client=client,
        #     messages=messages,
        #     model=model_name,
        #     max_tokens=max_tokens,
        #     temperature=temperature,
        #     top_p=top_p,
        #     seed=seed,
        #     n=1,
        #     max_retries=max_retries,
        #     retry_delay=retry_delay,
        #     timeout=60
        # )
        response = client.generate(
            prompt=prompt,
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            n=1,
            timeout=60
        )
        item['synthesis_result']['gpt_response'] = response[0]
        results.append(item)

    append_jsonl(cache_file, [item['synthesis_result'] for item in results])
    return results

FILE_NAME = Path(__file__).stem
default_output_dir = Path(__file__).parent / "outputs"
def main(
    dataset_name: str = "ise-uiuc/Magicoder-Evol-Instruct-110K",
    max_samples: Optional[int] = 5,
    model_name: str = "gpt-4.1-mini",
    max_tokens: int = 8192,
    top_p: float = 0.95,
    temperature: float = 0.6,
    seed: int = 42,
    n: int = 1,
    num_proc: int = 1,
    output_dir: str = "outputs",
    overwrite: bool = False,
    previous_result_file: Optional[str] = None,
    save_batch_size: int = 20,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    api_key: Optional[str] = None,
    base_url: str = "https://api.openai.com/v1",
    generation_mode: str = "questions_and_tests",  # "questions_and_tests", "programs", "test_cases"
):
    prev_tests_dict = load_previous_tests(previous_result_file) if previous_result_file else {}
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        print(f'---------the selected api_key is {api_key}---------')
        if api_key is None:
            raise ValueError("OpenAI API key not found.")

    print(f"Processing dataset: {dataset_name}")
    print(f"Model: {model_name}")
    print(f"Max samples: {max_samples}")
    print(f"Batch size: {save_batch_size}")

    output_dir = Path(output_dir) / pretty_name(dataset_name) / pretty_name(model_name)
    cache_file = output_dir / "step1_prompting.cache.jsonl"
    output_file = output_dir / "step1_prompting_results.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_file.exists() and not overwrite:
        print(f"Output file {output_file} already exists. Use --overwrite to overwrite.")
        return

    dataset = preprocess_dataset(dataset_name, max_sample=max_samples, num_proc=num_proc)
    data = list(dataset)

    cached_data = {}
    if cache_file.exists():
        print(f"Loading existing cache from {cache_file}")
        existing_cache = load_jsonl(cache_file)
        cached_data = {item['hash_id']: item for item in existing_cache}
        print(f"Loaded {len(cached_data)} cached items")

    items_to_process, items_to_process_map, final_results = [], {}, []
    for item in data:
        messages = [{"role": "user", "content": PROMPT_TEMPLATE_RAW.format(program=item['program'], instruction=item.get('problem', ''), test_cases="")}] if item.get('problem') else [{"role": "user", "content": PROMPT_TEMPLATE_NO_INSTRUCTION.format(program=item['program'])}]
        hash_id = hash_messages(messages)
        item['synthesis_result'] = {"hash_id": hash_id}
        if hash_id in cached_data:
            item['synthesis_result']['gpt_response'] = cached_data[hash_id]['gpt_response']
        else:
            items_to_process.append(item)
            items_to_process_map[hash_id] = len(final_results)
        final_results.append(item)

    if not items_to_process:
        print("All items are cached, saving final results...")
        with open(output_file, 'w') as f:
            for item in final_results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Results saved to {output_file}")
        return

    client = OpenAISyncClient(api_key=api_key, base_url=base_url)
    for i in range(0, len(items_to_process), save_batch_size):
        batch_items = items_to_process[i:i + save_batch_size]
        batch_results = process_batch_sync(
            client=client,
            batch_items=batch_items,
            model_name=model_name,
            max_tokens=max_tokens,
            cache_file=cache_file,
            previous_tests_dict=prev_tests_dict,
            generation_mode=generation_mode,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        for result_item in batch_results:
            idx = items_to_process_map[result_item['synthesis_result']['hash_id']]
            final_results[idx] = result_item

    with open(output_file, 'w') as f:
        for item in final_results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Processed dataset saved to {output_file}")
    if cache_file.exists():
        os.remove(cache_file)
        print(f"Cache file {cache_file} removed")


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