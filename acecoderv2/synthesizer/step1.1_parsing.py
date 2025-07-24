import os
import datasets
import json
from typing import List, Optional, Tuple
from fire import Fire
from tqdm import tqdm
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

FILE_NAME = Path(__file__).stem
default_output_dir = Path(__file__).parent / "outputs" / FILE_NAME
def main(
    file_path: str,
    num_proc: int = 1,
    output_dir: str = None,
):
    """
    Main function to generate test cases for a given dataset.
    :param dataset_name: Name of the dataset to process.
    :param ct: Number of test cases to generate.
    """
    assert os.path.exists(file_path), f"File {file_path} does not exist."
    
    output_dir = Path(output_dir) if output_dir else default_output_dir
    output_file = output_dir / f"{Path(file_path).stem}.jsonl"

    dataset = datasets.Dataset.from_json(file_path)

    def parsing_item(item):
        gpt_response = item.get('gpt_response', None)
        
        # parse the response
        try:
            obj = parse_incomplete_json(gpt_response)
            question = obj.get("question", "please ignore this question")
            question = json.dumps(question, ensure_ascii=False) if not isinstance(question, str) else question
            tests = obj.get("tests", ["assert False"])
        except Exception as e:
            print(f"Error parsing response: {e}")
            question = "Error in question generation"
            tests = ["assert False"]
        
        item['gpt_question'] = question
        item['tests'] = tests
        print("gpt_question:", type(item['gpt_question']))
        print("tests:", type(item['tests']))
        return item
    
    # Process the dataset in parallel
    dataset = dataset.map(
        parsing_item,
        num_proc=num_proc,
        desc="Parsing dataset",
    )
    # Save the processed dataset
    dataset.to_json(output_file, orient="records", lines=True)
    print(f"Processed dataset saved to {output_file}")

if __name__ == "__main__":
    Fire(main)

"""
This code is part of the AceCoderV2 project, which is designed to generate challenging LeetCode-style questions and test cases from code snippets using OpenAI's GPT models. The main function orchestrates the preprocessing of datasets, generation of test cases, and saving the results to a specified output directory. It supports parallel processing for efficiency and allows for caching of previous responses to avoid redundant API calls.
Usage:
```bash
python step1.1_parsing.py --file_path outputs/step1_prompting/Magicoder_Evol_Instruct_110K_gpt_4o_mini.jsonl --num_proc 1
```
"""