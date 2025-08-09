import fire
import torch
import json
import os
from pathlib import Path
from typing import Optional
from vllm import LLM, SamplingParams
from tqdm import tqdm
from typing import List
from transformers import AutoTokenizer
from acecoderv2.synthesizer.utils import pretty_name, append_jsonl, hash_messages

def load_vllm_model(
    model_name_or_path: str,
    torch_dtype: Optional[torch.dtype] = torch.bfloat16,
    tensor_parallel_size: int = 1,
    **kwargs,
):
    print("load model from %s" % model_name_or_path)
    print("torch_dtype:", torch_dtype)
    print("tensor_parallel_size:", tensor_parallel_size)
    print("kwargs:", kwargs)
    model_vllm = LLM(model_name_or_path, dtype=torch_dtype, tensor_parallel_size=tensor_parallel_size, **kwargs)
    return model_vllm

def preprocess_prompts_auto(
    data: List[dict],
    tokenizer: AutoTokenizer,
):
    """
    Preprocess prompts using the AutoTokenizer.
    """
    prompts = []
    for item in data:
        messages = [{"role": "user", "content": item['synthesis_result']["problem"]}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
    return prompts

def preprocess_prompts(data: List[dict], tokenizer: AutoTokenizer, mode:str="auto") -> List[str]:
    if mode == "auto":
        return preprocess_prompts_auto(data, tokenizer)
    else:
        raise ValueError(f"Unsupported mode: {mode}. Supported modes: 'auto'.")


# FILE_NAME = Path(__file__).stem
FILE_NAME = "step2.1_gen"

def main(
    file_path: str,
    output_dir: str = None,
    cache_dir: str = None,
    start_idx = None,
    end_idx: Optional[int] = None,
    save_batch_size: int = 16,
    model_name_or_path: str = "meta-llama/Llama-2-7b-chat-hf",
    torch_dtype: Optional[str] = "bfloat16",
    tensor_parallel_size: int = 1,
    seed: int = 42,
    top_p: float = 0.95,
    top_k: int = 1,
    n=1,
    temperature: float = 0.6,
    max_tokens: int = 32768,
    overwrite: bool = False,
    device_id: str = None,
    **vllm_kwargs
):
    if device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        print(f"Using CUDA_VISIBLE_DEVICES={device_id}")
    
    output_dir = Path(output_dir) if output_dir else Path(file_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if start_idx is None and end_idx is None:
        cache_file = output_dir / f"{FILE_NAME}_{pretty_name(model_name_or_path)}_seed{seed}.cache.jsonl"
        output_file = output_dir / f"{FILE_NAME}_{pretty_name(model_name_or_path)}_seed{seed}.jsonl"
    else:
        if end_idx is None:
            end_idx = len(data)
        cache_file = output_dir / f"{FILE_NAME}_{pretty_name(model_name_or_path)}_seed{seed}_{start_idx}_{end_idx}.cache.jsonl"
        output_file = output_dir / f"{FILE_NAME}_{pretty_name(model_name_or_path)}_seed{seed}_{start_idx}_{end_idx}.jsonl"

    if output_file.exists() and output_file.stat().st_size != 0 and not overwrite:
        print(f"Output file {output_file} already exists. Use --overwrite to overwrite.")
        return
    
    # Load cached data if exists
    cached_data = {}
    if cache_file.exists() and not overwrite:
        with open(cache_file, 'r') as f:
            for line in f.readlines():
                if line.strip():
                    item = json.loads(line)
                    cached_data[item['qid']] = item
        print(f"Loaded {len(cached_data)} cached items from {cache_file}")
    
    # Load input data
    if file_path.endswith('.jsonl'):
        with open(file_path, 'r') as f:
            data = [json.loads(line) for line in f.readlines()]
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
    else:
        raise ValueError("Unsupported file format. Please provide a .jsonl or .json file.")

    if start_idx is not None and end_idx is not None:
        data = data[start_idx:end_idx]
    
    print(f"Processing {len(data)} items from {start_idx} to {end_idx}...")

    # Identify items that need processing (not in cache)
    items_to_process = []
    final_results = []
    
    for item in data:
        qid = hash_messages(item['synthesis_result']['problem'])
        if qid in cached_data:
            # Use cached result
            new_item = item.copy()
            new_item['gen_result'] = cached_data[qid]
        else:
            new_item = item.copy()
            new_item['gen_result'] = {
                'outputs': [],
                'qid': qid,
                'prompt': None,
                'sampling_params': {
                    "model_name_or_path": model_name_or_path,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "n": n,
                    "max_tokens": max_tokens,
                    "seed": seed,
                }
            }
            # Needs processing
            items_to_process.append(new_item)
        final_results.append(new_item)  # Will be updated with results later

    print(f"Found {len(cached_data)} cached items, {len(items_to_process)} items need processing")
    
    if len(items_to_process) == 0:
        print("All items are cached, saving final results...")
        # Save final results
        with open(output_file, 'w') as f:
            for item in final_results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Results saved to {output_file}")
        return

    # Load model and tokenizer only if we have items to process
    torch_dtype = getattr(torch, torch_dtype, torch.bfloat16)
    model_vllm = load_vllm_model(
        model_name_or_path=model_name_or_path,
        torch_dtype=torch_dtype,
        tensor_parallel_size=tensor_parallel_size,
        **vllm_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # Preprocess prompts for items that need processing
    prompts = preprocess_prompts(items_to_process, tokenizer)
    qids = [item['gen_result']['qid'] for item in items_to_process]

    # Set up sampling parameters
    if top_p < 1:
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
            n=n
        )
    else:
        sampling_params = SamplingParams(
            temperature=temperature,
            top_k=top_k,
            max_tokens=max_tokens,
            seed=seed,
            n=n
        )
    
    # Generate responses in batches
    processed_items = []
    qid_to_result_idx = {item['gen_result']['qid']: idx for idx, item in enumerate(final_results)}
    
    for i in tqdm(range(0, len(prompts), save_batch_size), desc="Generating responses"):
        batch_prompts = prompts[i:i + save_batch_size]
        batch_qids = qids[i:i + save_batch_size]
        batch_items = items_to_process[i:i + save_batch_size]
        
        outputs = model_vllm.generate(batch_prompts, sampling_params=sampling_params)
        
        batch_results = []
        for j, output in enumerate(outputs):
            generated_texts = [output.outputs[k].text.strip() for k in range(len(output.outputs))]

            # Update the item with results
            batch_items[j]['gen_result']['outputs'].extend(generated_texts)
            batch_items[j]['gen_result']['prompt'] = batch_prompts[j]
            
            batch_results.append(batch_items[j]['gen_result'])
            processed_items.append(batch_items[j])
            
            # Update final results
            result_idx = qid_to_result_idx[batch_qids[j]]
            final_results[result_idx] = batch_items[j]
        
        # Save batch to cache
        append_jsonl(cache_file, batch_results)
        
        print(f"Saved batch {i//save_batch_size + 1} to cache ({len(batch_results)} items)")

    print(f"Generated responses for {len(processed_items)} items")

    # Save final results
    with open(output_file, 'w') as f:
        for item in final_results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Final results saved to {output_file}")
    
    # Remove cache file
    if cache_file.exists():
        os.remove(cache_file)
        print(f"Cache file {cache_file} removed")

if __name__ == "__main__":
    fire.Fire(main)

"""
python step2.1_vllm_gen.py outputs/Magicoder_Evol_Instruct_110K/gpt_4o_mini/step1.1_parsing.jsonl \
    --start_idx=0 \
    --end_idx=50 \
    --save_batch_size=16 \
    --model_name_or_path='Qwen/Qwen2.5-Coder-7B-Instruct' \
    --tensor_parallel_size=1 \
    --top_p=0.95 --top_k=1 --temperature=0.6 --max_tokens=2048 --n 8
"""