import os
import fire
import datasets
import json
from pathlib import Path
from acecoderv2.synthesizer.utils import hash_messages

def pretty_name(name: str) -> str:
    """
    Convert a name to a pretty name by extracting the last part after '/' and replacing '-' with '_'.
    
    Args:
        name (str): The original model or dataset name/path
        
    Returns:
        str: A cleaned name with last part after '/' and '-' replaced with '_'
    """
    # Extract part after last '/'
    name = name.split('/')[-1]
    # Replace '-' with '_'
    name = name.replace('-', '_')
    return name

def format_step1_prompting_data(item):
    return {
        "id": item['id'],
        "problem": item['ori_question'],
        "response": None,
        "program": item['ori_program'],
        "synthesis_result": {
            "hash_id": hash_messages(item['question']),
            "gpt_response": json.dumps({"question": item['question'], "tests": item['tests']}, ensure_ascii=False),
            "problem": item['question'],
            "tests": item['tests'],
        }
    }

def format_step2_gen_data(item, model_name:str):
    model_inferences = [x for x in item['inferences'] if x.get('model') == model_name]
    hash_id = hash_messages(item['question'])
    return {
        "id": item['id'],
        "problem": item['ori_question'],
        "response": None,
        "program": item['ori_program'],
        "synthesis_result": {
            "hash_id": hash_id,
            "gpt_response": json.dumps({"question": item['question'], "tests": item['tests']}, ensure_ascii=False),
            "problem": item['question'],
            "tests": item['tests'],
        },
        "gen_result": {
            "outputs": [x['inference'] for x in model_inferences],
            "qid": item['id'],
            "prompt": None,
            "sampling_params": {
                "model_name_or_path": model_name,
                "n": len(model_inferences),
            }
        },
    }

def main(
    dataset_path="CodeDPO/AceCoderV2-150K-processed",
    output_dir="outputs"
):
    output_dir = Path(output_dir)
    output_dir = output_dir / pretty_name(dataset_path) / pretty_name("o1-mini")
    output_dir.mkdir(parents=True, exist_ok=True)
    step_1_output_file = output_dir / "step1_prompting_results.jsonl"

    dataset = datasets.load_dataset(dataset_path, split="train")
    
    step_1_dataset = dataset.map(
        format_step1_prompting_data,
        desc="Formatting step 1 prompting data",
        num_proc=32,
        remove_columns=dataset.column_names
    )

    with open(step_1_output_file, 'w') as f:
        for item in step_1_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Step 1 prompting data saved to {step_1_output_file}")

    step_1_1_output_file = output_dir / "step1.1_parsing.jsonl"
    # set this to symbolic link to step1_prompting_results.jsonl
    if step_1_1_output_file.exists():
        step_1_1_output_file.unlink()
    step_1_1_output_file.symlink_to(step_1_output_file.name)
    print(f"Step 1.1 parsing data linked to {step_1_output_file}")

    # step2.1 gen
    available_models = set([x['model'] for x in dataset[0]['inferences'] if 'model' in x])
    for model in available_models:
        print(f"Available model: {model}")
        step_2_output_file = output_dir / f"step2.1_gen_{pretty_name(model)}_seed42.jsonl"
        step_2_dataset = dataset.map(
            lambda item: format_step2_gen_data(item, model),
            desc=f"Formatting step 2.1 gen data for {model}",
            num_proc=32,
            remove_columns=dataset.column_names
        )
        # filter those without inferences
        step_2_dataset = step_2_dataset.filter(lambda item: len(item['gen_result']['outputs']) > 0, num_proc=32, desc=f"Filtering items without inferences for {model}")
        print(f"Before filtering, {len(dataset)} items, after filtering, {len(step_2_dataset)} items for model {model}")
        with open(step_2_output_file, 'w') as f:
            for item in step_2_dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Step 2.1 gen data for {model} saved to {step_2_output_file}")
        print(f"Processed {len(step_2_dataset)} items for model {model}")

if __name__ == "__main__":
    fire.Fire(main)