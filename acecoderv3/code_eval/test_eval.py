# test
# 1. test prime code compute score
from acecoderv2.code_eval import eval_codes, prime_code_compute_score_async
import datasets
# dataset = datasets.load_dataset("agentica-org/DeepCoder-Preview-Dataset", 'taco', split='train')
# dataset = dataset.select(range(50))
# solution_strs = [item['solutions'][-1] for item in dataset]
# test_cases = [item['tests'] for item in dataset]
# scores = eval_codes(solution_strs, test_cases, return_test_cases_pass_status=True)
# print(scores)

# print(prime_code_compute_score_async(solution_strs[2], test_cases[2]))
# # for i, score in enumerate(scores):
# #     if score != 1:
# #         print(f"Solution {i} failed with score: {score}")
# #         print(f"Solution: {solution_strs[i]}")
# #         print(f"Test cases: {test_cases[i]}")



dataset = datasets.load_dataset("TIGER-Lab/AceCode-87K", split='train')
dataset = dataset.select(range(50))
solution_strs = [item['inferences'][0]['completion'] for item in dataset]
expected_pass_rates = [item['inferences'][0]['pass_rate'] for item in dataset]
test_cases = [item['test_cases'] for item in dataset]
scores, test_case_pass_status = eval_codes(solution_strs, test_cases, return_test_cases_pass_status=True)
for i, score in enumerate(scores):
    print(f"Solution {i} score: {score}, expected: {expected_pass_rates[i]}, pass status: {test_case_pass_status[i]}")