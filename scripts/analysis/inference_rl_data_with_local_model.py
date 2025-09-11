import pandas as pd
import json
from tqdm import tqdm
from synthesizer.utils import load_jsonl, save_jsonl
from code_eval import eval_codes
import matplotlib.pyplot as plt


INFERENCE_PATH = "saved_inferences_step1.jsonl"
INFERENCE_WITH_EVAL_PATH = "saved_inferences_step2.jsonl"


def load_parquet(data_path: str):
    df = pd.read_parquet(data_path)
    prompt_lst = []
    test_lst = []
    def extract_prompt_and_tests(entry):
        prompt = entry["prompt"]
        tests = entry["extra_info"]["test_cases"]
        return prompt, tests
    for _, entry in df.iterrows():
        prompt, tests = extract_prompt_and_tests(entry)
        prompt_lst.append(prompt)
        test_lst.append(tests)
    return prompt_lst, test_lst

def create_inference(data_path: str):
    from vllm import SamplingParams, LLM
    prompt_lst, test_lst = load_parquet(data_path=data_path)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2048)
    llm = LLM(model="Qwen/Qwen2.5-Coder-7B-Instruct")
    for i, msg in enumerate(tqdm(prompt_lst)):
        output = llm.chat(messages=msg, sampling_params=sampling_params, use_tqdm=False)
        with open(INFERENCE_PATH, "a") as f:
            f.write(json.dumps({"idx": i, "response": output[0].outputs[0].text}) + "\n")

def evaluate_inference(data_path: str):
    prompt_lst, test_lst = load_parquet(data_path=data_path)
    inferences = load_jsonl(INFERENCE_PATH)
    def extract_python_program(program: str):
        idx = program.find("```python")
        if idx < 0:
            return None
        program = program[idx + 9 :]
        idx = program.find("```")
        if idx < 0:
            None
        program = program[:idx]
        return program
    
    out = []
    programs = []
    tests = []
    ids = []
    for inference in tqdm(inferences):
        _id, prog = inference["idx"], extract_python_program(inference["response"])
        if prog is not None:
            test = test_lst[_id].tolist()
            programs.append(prog)
            ids.append(_id)
            tests.append(test)

    eval_results = eval_codes(solution_strs=programs, test_cases=tests)
    for prog, test, id, acc in zip(programs, tests, ids, eval_results):
        out.append({"idx": id, "pass_rate": acc, "program": prog, "test": test})

    save_jsonl(INFERENCE_WITH_EVAL_PATH, out)
            
def create_graph():
    lst = load_jsonl(INFERENCE_WITH_EVAL_PATH)
    pass_rate = [i["pass_rate"] * 100 for i in lst]
    plt.hist(pass_rate, bins=50, edgecolor='black')

    # Add labels and title
    plt.xlabel('Pass Rate (%)')
    plt.ylabel('Frequency')
    plt.title('Frequency Plot of Pass Rate')
    plt.savefig("frequency_plot.jpg", dpi=300, bbox_inches='tight')

    
if __name__ == "__main__":
    data_path = "/home/wyett/verl-tool/data/acecoder/r1/train.parquet"
    # create_inference(data_path)
    # evaluate_inference(data_path)
    create_graph()

    