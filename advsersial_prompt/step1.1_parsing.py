import os
import datasets
import json
from typing import List, Optional, Tuple
from fire import Fire
from tqdm import tqdm
from pathlib import Path
import sys
from pathlib import Path

from utils import (
    parse_incomplete_json,
    append_jsonl,
    load_jsonl,
    chunking,
    get_python_code_from_string,
    hash_messages,
    pretty_name
)

FILE_NAME = Path(__file__).stem
ERROR_QUESTION = "Error in question generation"
ERROR_TESTS = ["assert False"]

def filter_parsed_items(item):
    """
    Filter function to check if the item has 'gpt_response' and 'tests'.
    """
    gpt_response = item['synthesis_result'].get('gpt_response', None)
    tests = item['synthesis_result'].get('tests', None)
    if gpt_response and gpt_response != ERROR_QUESTION and tests and tests != ERROR_TESTS:
        return True
    return False

def main(
    file_path: str,
    num_proc: int = 1,
    output_dir: str = None,
    do_filter: bool = True,
    overwrite: bool = False,
    parsing_mode: str = "questions_and_tests"  # "questions_and_tests", "programs", "test_cases"
):
    """
    Main function to generate test cases for a given dataset.
    :param dataset_name: Name of the dataset to process.
    :param ct: Number of test cases to generate.
    """
    from pathlib import Path

    assert os.path.exists(file_path), f"File {file_path} does not exist."

    output_dir = Path(output_dir) if output_dir else Path(file_path).parent

    output_file = output_dir / f"{FILE_NAME}.jsonl"

    if output_file.exists() and output_file.stat().st_size != 0 and not overwrite:
        print(f"Output file {output_file} already exists and is not empty. Use --overwrite to overwrite.")
        return

    dataset = datasets.Dataset.from_json(file_path)
    # def parsing_item(item):
    #     gpt_response = item['synthesis_result'].get('gpt_response', None)
    #     print(f'---------the gpt_response is {gpt_response}---------')
    #     # parse the response
    #     try:
    #         obj = parse_incomplete_json(gpt_response)
    #         question = obj.get("question", ERROR_QUESTION)
    #         question = json.dumps(question, ensure_ascii=False) if not isinstance(question, str) else question
    #         tests = obj.get("tests", ERROR_TESTS)
    #     except Exception as e:
    #         print(f"Error parsing response: {e}")
    #         question = ERROR_QUESTION
    #         tests = ERROR_TESTS
        
    #     item['synthesis_result']['problem'] = question
    #     item['synthesis_result']['tests'] = tests
    #     # print("gpt_question:", type(item['gpt_question']))
    #     # print("tests:", type(item['tests']))
    #     return item
    
    # def parsing_item(item):
    #     gpt_response = item['synthesis_result'].get('gpt_response', None)
    #     print(f'---------the gpt_response is {gpt_response}---------')

    #     try:
    #         # 🔽 新增：从 dict 中提取 GPT 返回的字符串
    #         if isinstance(gpt_response, dict):
    #             raw_text = gpt_response.get("message", {}).get("content", "")
    #         elif isinstance(gpt_response, str):
    #             raw_text = gpt_response
    #         else:
    #             raw_text = str(gpt_response)

    #         # 🔽 再传给解析器（原始 JSON 格式字符串）
    #         obj = parse_incomplete_json(raw_text)

    #         question = obj.get("question", ERROR_QUESTION)
    #         question = json.dumps(question, ensure_ascii=False) if not isinstance(question, str) else question
    #         tests = obj.get("tests", ERROR_TESTS)
    #     except Exception as e:
    #         print(f"Error parsing response: {e}")
    #         question = ERROR_QUESTION
    #         tests = ERROR_TESTS
        
    #     item['synthesis_result']['problem'] = question
    #     item['synthesis_result']['tests'] = tests
    #     return item
    
    def parsing_item(item):
        gpt_response = item['synthesis_result'].get('gpt_response', None)
        # Process GPT response

        # 处理 gpt_response 为字典或字符串的情况
        if isinstance(gpt_response, dict):
            raw_text = gpt_response.get("message", {}).get("content", "")
        else:
            raw_text = str(gpt_response)
        
        # Parse JSON content from GPT response

        try:
            if parsing_mode == "questions_and_tests":
                # Original behavior: parse questions and test cases
                obj = parse_incomplete_json(raw_text)
                question = obj.get("question", ERROR_QUESTION)
                tests = obj.get("tests", ERROR_TESTS)
                item['synthesis_result']['problem'] = question
                item['synthesis_result']['tests'] = tests
            elif parsing_mode == "programs":
                # Parse generated programs
                obj = parse_incomplete_json(raw_text)
                programs = obj.get("programs", [])
                if not programs:
                    # Fallback: try to extract code blocks from raw text
                    import re
                    code_blocks = re.findall(r'```python\n(.*?)\n```', raw_text, re.DOTALL)
                    if not code_blocks:
                        code_blocks = re.findall(r'```\n(.*?)\n```', raw_text, re.DOTALL)
                    programs = [block.strip() for block in code_blocks if block.strip()]
                
                item['synthesis_result']['generated_programs'] = programs
                # Store parsed programs
            elif parsing_mode == "test_cases":
                # Parse test cases (similar to step2.1 logic)
                test_cases = []
                if "assert" in raw_text:
                    # Extract assert statements
                    import re
                    asserts = re.findall(r'assert [^\\n]+', raw_text)
                    test_cases = asserts
                else:
                    # Try JSON parsing
                    obj = parse_incomplete_json(raw_text)
                    test_cases = obj.get("tests", obj.get("test_cases", []))
                
                item['synthesis_result']['generated_test_cases'] = test_cases
                # Store parsed test cases
            else:
                raise ValueError(f"Unknown parsing_mode: {parsing_mode}")
                
        except Exception as e:
            print(f"⚠️  Parsing error: {str(e)[:100]}...")
            if parsing_mode == "questions_and_tests":
                item['synthesis_result']['problem'] = ERROR_QUESTION
                item['synthesis_result']['tests'] = ERROR_TESTS
            elif parsing_mode == "programs":
                item['synthesis_result']['generated_programs'] = []
            elif parsing_mode == "test_cases":
                item['synthesis_result']['generated_test_cases'] = []
        
        return item
    # Process the dataset in parallel
    dataset = dataset.map(
        parsing_item,
        num_proc=num_proc,
        desc="Parsing dataset",
    )
    if do_filter:
        print(f"Before filtering, dataset size: {len(dataset)}")
        dataset = dataset.filter(
            filter_parsed_items,
            num_proc=num_proc,
            desc="Filtering parsed items",
        )
        print(f"After filtering, dataset size: {len(dataset)}")
    # Save the processed dataset
    dataset.to_json(output_file, orient="records", lines=True)
    print(f"Processed dataset saved to {output_file}")

if __name__ == "__main__":
    Fire(main)

"""
This code is part of the AceCoderV2 project, which is designed to generate challenging LeetCode-style questions and test cases from code snippets using OpenAI's GPT models. The main function orchestrates the preprocessing of datasets, generation of test cases, and saving the results to a specified output directory. It supports parallel processing for efficiency and allows for caching of previous responses to avoid redundant API calls.
Usage:
```bash
python step1.1_parsing.py --file_path outputs/Magicoder_Evol_Instruct_110K/gpt_4o_mini/step1_prompting_results.jsonl --num_proc 1
```
"""