import json
import asyncio
import psutil
from .prime_code import compute_score as prime_code_compute_score
from .acecoder import evaluate_test_cases
from .utils import parse_code, hash_string, check_syntax
from tqdm.asyncio import tqdm
from typing import Optional, List, Tuple, Union
from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor

import warnings

# Suppress multiprocessing cleanup warnings
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing")
DEFAULT_NUM_PROCESSES = 64


def prime_code_compute_score_async(solution_str:str, test_cases:Union[str, dict]):
    try:
        if not isinstance(test_cases, dict):
            test_cases = json.loads(test_cases)
    except Exception as e:
        print(f"Error:{e}")
    pass_rate, meta_data_list = prime_code_compute_score(solution_str, test_cases, continuous=True)
    pass_rate = float(pass_rate)
    if pass_rate == 1:
        test_case_pass_status = [True for _ in range(len(test_cases['inputs']))]
    else:
        test_case_pass_status = []
        for meta_data in meta_data_list:
            res = json.loads(meta_data['test_case']['res'])
            print(f"res: {res}")
            assert len(res) == 1, "Expected single test case result"
            test_case_pass_status.append(res[0] == True)
    return pass_rate, test_case_pass_status

    

async def single_compute_score(evaluation_func, completion, test_cases, executor, timeout=300.0):
    loop = asyncio.get_running_loop()
    try:
        # Ensure process_completion is called properly
        future = loop.run_in_executor(executor, partial(evaluation_func, completion, test_cases))
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        print(f"[Timeout] Task timeout: {completion}")
        return None  # Default value for timed-out rows
    except Exception as e:
        print(f"[Error] Task failed: {e}, completion: {completion[:80]}")
        return None  # Default value for failed rows


async def parallel_compute_score_async(
    evaluation_func, completions:List[str], test_cases: List[Union[str, dict]], num_processes=DEFAULT_NUM_PROCESSES
):
    scores = []
    test_cases_pass_status = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # to prevent very occasional starvation caused by some anomalous programs ( like infinite loop ), the
        # exceptions in async programs will instantly halt the evaluation, and all summoned processes will be killed.
        try:
            # Create tasks for all rows
            tasks_async = [
                single_compute_score(evaluation_func, c, t, executor, timeout=300.0)
                for c, t in zip(completions, test_cases)
            ]
            results = await tqdm.gather(*tasks_async)
        except Exception as e:
            print(f"[Exception] async gather failed: {e}")
            raise
        finally:
            terminated_count = 0
            for pid, proc in executor._processes.items():
                try:
                    p = psutil.Process(pid)
                    p.terminate()
                    try:
                        p.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        p.kill()
                    terminated_count += 1
                except Exception:
                    pass
            print(f"[Shutdown] {terminated_count} subprocess(es) terminated.")

    # Process results
    for result, completion, reference in zip(results, completions, test_cases, strict=True):
        if isinstance(result, Exception) or result is None:
            # Handle failed or timed-out tasks
            scores.append(0.0)
            test_cases_pass_status.append([False]) # 1 dummy
        elif isinstance(result, int | float | bool):
            scores.append(float(result))
        else:
            scores.append(float(result[0]))
            test_cases_pass_status.append(result[1])
    return scores, test_cases_pass_status

def get_prime_code_data_score(solution_strs:List[str], test_cases: List[Union[str, dict]], binary:bool=False, num_processes=DEFAULT_NUM_PROCESSES):
    """ Get the scores for the given solutions and test cases using Prime Code evaluation.
    Args:
        solution_strs: List of solution strings.
        test_cases: List of test case strings (should be lists of assert statements).
        binary: If True, return binary scores (1.0 or -1.0),
        num_processes: Number of processes to use for parallel computation.
    Returns:
        List of scores for each solution.
        If binary is True, scores will be 1.0 for passing solutions and 0.0 for failing solutions.
        If binary is False, scores will be the pass rates.
    """
    scores = [0. for _ in range(len(solution_strs))]
    pass_rates, test_cases_pass_status = asyncio.run(
        parallel_compute_score_async(
            prime_code_compute_score_async,
            solution_strs,
            test_cases,
            num_processes=num_processes
        )
    ) # list of 1.0 or 0.0
    for i in range(len(scores)):
        if binary:
            scores[i] = 1.0 if pass_rates[i] == 1.0 else 0.0
        else:
            scores[i] = pass_rates[i]
    test_cases_pass_status = [
        {'pass': pass_status, 'reason': None, 'error_message': None}
        for pass_status in test_cases_pass_status
    ]
    return scores, test_cases_pass_status

def get_acecoder_data_score(solution_strs: List[str], test_cases: List[Union[str, list]], binary: bool = False, num_processes: int = DEFAULT_NUM_PROCESSES):
    """ Get the scores for the given solutions and test cases using AceCoder evaluation.
    Args:
        solution_strs: List of solution strings.
        test_cases: List of test case strings (should be lists of assert statements).
        binary: If True, return binary scores (1.0 or -1.0),
        num_processes: Number of processes to use for parallel computation.
    Returns:
        List of scores for each solution.
        If binary is True, scores will be 1.0 for passing solutions and 0.0 for failing solutions.
        If binary is False, scores will be the pass rates.
    """
    samples = [
        {
            'task_id': hash_string(solution_str),
            'output': solution_str,
            'tests': json.loads(test_case) if isinstance(test_case, str) else test_case,
            '_identifier': i,
        }
        for i, (solution_str, test_case) in enumerate(zip(solution_strs, test_cases))
    ]
    results, pass_dates = evaluate_test_cases(
        samples, n_workers=num_processes, extract_solution=True, test_details=True, i_just_wanna_run=True, min_time_limit=1, gt_time_limit_factor=1
    )

    if binary:
        scores = [1.0 if res['eval_results']['pass_rate'] == 1.0 else 0 for res in results]
    else:
        scores = [res['eval_results']['pass_rate'] for res in results]
    test_cases_pass_status = [
        [detail for detail in res['eval_results']['details']] for res in results
    ]
    return scores, test_cases_pass_status

def eval_codes(solution_strs:List[str], test_cases: List[Union[str, dict, list]], num_processes: int = DEFAULT_NUM_PROCESSES, binary: bool = False, return_test_cases_pass_status: bool = False):
    test_cases = [json.loads(test_case) if isinstance(test_case, str) else test_case for test_case in test_cases]
    prime_idxs = []
    acecoder_idxs = []
    for idx in range(len(solution_strs)):
        if isinstance(test_cases[idx], list):
            acecoder_idxs.append(idx)
        else:
            prime_idxs.append(idx)
    print(f"Prime Code samples: {len(prime_idxs)}, AceCoder samples: {len(acecoder_idxs)}")
    prime_scores = []
    acecoder_scores = []
    if prime_idxs:
        prime_scores, test_cases_pass_status = get_prime_code_data_score(
            [solution_strs[idx] for idx in prime_idxs],
            [test_cases[idx] for idx in prime_idxs],
            binary=binary, num_processes=num_processes
        )
    if acecoder_idxs:
        acecoder_scores, test_cases_pass_status = get_acecoder_data_score(
            [solution_strs[idx] for idx in acecoder_idxs],
            [test_cases[idx] for idx in acecoder_idxs],
            binary=binary, num_processes=num_processes
        )
    scores = [0.0] * len(solution_strs)
    for idx, score in zip(prime_idxs, prime_scores):
        scores[idx] = score
    for idx, score in zip(acecoder_idxs, acecoder_scores):
        scores[idx] = score

    if return_test_cases_pass_status:
        return scores, test_cases_pass_status
    else:
        return scores
