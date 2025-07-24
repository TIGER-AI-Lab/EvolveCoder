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
from acecoderv2.synthesizer.utils import pretty_name, append_jsonl

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
        messages = [{"role": "user", "content": item["problem"]}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
    return prompts, [item["hash_id"] for item in data]

def preprocess_prompts(data: List[dict], tokenizer: AutoTokenizer, mode:str="auto") -> List[str]:
    if mode == "auto":
        return preprocess_prompts_auto(data, tokenizer)
    else:
        raise ValueError(f"Unsupported mode: {mode}. Supported modes: 'auto'.")


# FILE_NAME = Path(__file__).stem
FILE_NAME = "step2.1_gen"
default_output_dir = Path(__file__).parent / "outputs" / FILE_NAME
default_cache_dir = Path(__file__).parent / "outputs" / FILE_NAME / "cache"

def main(
    file_path: str,
    output_dir: str = None,
    cache_dir: str = None,
    start_idx = 0,
    end_idx: Optional[int] = None,
    save_batch_size: int = 16,
    model_name_or_path: str = "meta-llama/Llama-2-7b-chat-hf",
    torch_dtype: Optional[str] = "bfloat16",
    tensor_parallel_size: int = 1,
    seed: int = 42,
    top_p: float = 0.95,
    top_k: int = 1,
    temperature: float = 0.6,
    max_tokens: int = 32768,
    overwrite: bool = False,
    device_id: str = None,
    **vllm_kwargs
):
    if device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        print(f"Using CUDA_VISIBLE_DEVICES={device_id}")
    
    output_dir = Path(output_dir) if output_dir else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(cache_dir) if cache_dir else default_cache_dir
    cache_file = cache_dir / Path(file_path).stem / f"{pretty_name(model_name_or_path)}_vllm_seed{seed}_{start_idx}_{end_idx}.jsonl"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load cached data if exists
    cached_data = {}
    if cache_file.exists() and not overwrite:
        with open(cache_file, 'r') as f:
            for line in f.readlines():
                if line.strip():
                    item = json.loads(line)
                    cached_data[item['hash_id']] = item
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

    if end_idx is None:
        end_idx = len(data)
    data = data[start_idx:end_idx]
    
    output_file = output_dir / Path(file_path).stem / f"{pretty_name(model_name_or_path)}_vllm_seed{seed}_{start_idx}_{end_idx}.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if output_file.exists() and not overwrite:
        print(f"Output file {output_file} already exists. Use --overwrite to overwrite.")
        return
    
    print(f"Processing {len(data)} items from {start_idx} to {end_idx}...")

    # Identify items that need processing (not in cache)
    items_to_process = []
    final_results = []
    
    for item in data:
        hash_id = item["hash_id"]
        if hash_id in cached_data:
            # Use cached result
            final_results.append(cached_data[hash_id])
        else:
            # Needs processing
            items_to_process.append(item)
            final_results.append(item)  # Will be updated with results later
    
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
    prompts, qids = preprocess_prompts(items_to_process, tokenizer)

    # Set up sampling parameters
    if top_p < 1:
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed
        )
    else:
        sampling_params = SamplingParams(
            temperature=temperature,
            top_k=top_k,
            max_tokens=max_tokens,
            seed=seed
        )
    
    # Generate responses in batches
    processed_items = []
    hash_id_to_result_idx = {item["hash_id"]: idx for idx, item in enumerate(final_results)}
    
    for i in tqdm(range(0, len(prompts), save_batch_size), desc="Generating responses"):
        batch_prompts = prompts[i:i + save_batch_size]
        batch_qids = qids[i:i + save_batch_size]
        batch_items = items_to_process[i:i + save_batch_size]
        
        outputs = model_vllm.generate(batch_prompts, sampling_params=sampling_params)
        
        batch_results = []
        for j, output in enumerate(outputs):
            response = output.outputs[0].text.strip()
            
            # Clean up response
            for end_token in ["<|im_end|>", "<|end_of_text|>", "<|eot_id|>"]:
                if end_token in response:
                    idx = response.index(end_token)
                    response = response[:idx]
                    break

            # Update the item with results
            result_item = batch_items[j].copy()
            result_item['llm_response'] = response
            result_item['qid'] = batch_qids[j]
            result_item['prompt'] = batch_prompts[j]
            result_item['sampling_params'] = {
                "model_name_or_path": model_name_or_path,
                "temperature": sampling_params.temperature,
                "top_p": sampling_params.top_p,
                "top_k": sampling_params.top_k,
                "max_tokens": sampling_params.max_tokens,
                "seed": sampling_params.seed,
            }
            
            batch_results.append(result_item)
            processed_items.append(result_item)
            
            # Update final results
            result_idx = hash_id_to_result_idx[batch_qids[j]]
            final_results[result_idx] = result_item
        
        # Save batch to cache
        for result_item in batch_results:
            append_jsonl(cache_file, result_item)
        
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
python step2.1_vllm_gen.py outputs/step1.1_parsing/Magicoder_Evol_Instruct_110K_gpt_4o_mini.jsonl \
    --start_idx=0 \
    --end_idx=50 \
    --save_batch_size=16 \
    --model_name_or_path='Qwen/Qwen2.5-Coder-7B-Instruct' \
    --tensor_parallel_size=2 \
    --top_p=0.95 --top_k=1 --temperature=0.6 --max_tokens=2048
"""