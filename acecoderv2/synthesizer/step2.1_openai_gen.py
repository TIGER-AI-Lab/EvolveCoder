import fire
import json
import os
import asyncio
import aiohttp
from pathlib import Path
from typing import Optional, List, Dict, Any
from tqdm.asyncio import tqdm
import time
from acecoderv2.synthesizer.utils import pretty_name, append_jsonl


class OpenAIAsyncClient:
    """
    Async OpenAI client using aiohttp for better control and performance.
    """
    
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300),  # 5 minute timeout
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 4000,
        seed: Optional[int] = None,
    ) -> str:
        """
        Send a chat completion request to OpenAI API.
        """
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        
        if seed is not None:
            payload["seed"] = seed
        
        async with self.session.post(url, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                return data["choices"][0]["message"]["content"]
            else:
                error_text = await response.text()
                raise aiohttp.ClientResponseError(
                    request_info=response.request_info,
                    history=response.history,
                    status=response.status,
                    message=f"OpenAI API error: {error_text}"
                )


def preprocess_prompts_auto(data: List[dict]) -> tuple[List[List[dict]], List[str]]:
    """
    Preprocess prompts for OpenAI API format.
    """
    messages_list = []
    for item in data:
        messages = [{"role": "user", "content": item["problem"]}]
        messages_list.append(messages)
    return messages_list, [item["id"] for item in data]


def preprocess_prompts(data: List[dict], mode: str = "auto") -> tuple[List[List[dict]], List[str]]:
    if mode == "auto":
        return preprocess_prompts_auto(data)
    else:
        raise ValueError(f"Unsupported mode: {mode}. Supported modes: 'auto'.")


async def generate_with_retry(
    client: OpenAIAsyncClient,
    messages: List[dict],
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    semaphore: asyncio.Semaphore = None,
) -> str:
    """
    Generate response with retry logic for handling rate limits and errors.
    Uses semaphore to limit concurrent requests.
    """
    async def _make_request():
        for attempt in range(max_retries):
            try:
                response = await client.chat_completion(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    seed=seed
                )
                return response
            except aiohttp.ClientResponseError as e:
                if e.status == 429:  # Rate limit
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        print(f"Rate limit hit, waiting {wait_time:.1f}s before retry {attempt + 2}/{max_retries}")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        return f"ERROR: Rate limit exceeded after {max_retries} attempts"
                elif e.status >= 500:  # Server error
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        print(f"Server error {e.status}, retrying in {wait_time:.1f}s (attempt {attempt + 2}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        return f"ERROR: Server error {e.status} after {max_retries} attempts"
                else:
                    return f"ERROR: HTTP {e.status} - {str(e)}"
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"Request timeout, retrying in {wait_time:.1f}s (attempt {attempt + 2}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    return f"ERROR: Timeout after {max_retries} attempts"
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"Unexpected error: {e}, retrying in {wait_time:.1f}s (attempt {attempt + 2}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    return f"ERROR: {str(e)}"
        
        return "ERROR: All retry attempts failed"
    
    if semaphore:
        async with semaphore:
            return await _make_request()
    else:
        return await _make_request()


async def process_batch(
    client: OpenAIAsyncClient,
    messages_batch: List[List[dict]],
    qids_batch: List[str],
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
    max_retries: int,
    retry_delay: float,
    max_concurrent: int,
) -> List[str]:
    """
    Process a batch of requests concurrently with controlled concurrency.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    tasks = []
    for messages in messages_batch:
        task = generate_with_retry(
            client=client,
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
            max_retries=max_retries,
            retry_delay=retry_delay,
            semaphore=semaphore
        )
        tasks.append(task)
    
    # Wait for all tasks to complete
    responses = await tqdm.gather(*tasks, desc="Processing batch", unit="request", total=len(tasks))
    return responses


# FILE_NAME = Path(__file__).stem
FILE_NAME = "step2.1_gen"
default_output_dir = Path(__file__).parent / "outputs" / FILE_NAME
default_cache_dir = Path(__file__).parent / "outputs" / FILE_NAME / "cache"


async def main_async(
    file_path: str,
    output_dir: str = None,
    cache_dir: str = None,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    batch_size: int = 20,  # Increased default batch size
    max_concurrent: int = 10,  # Increased default concurrency
    model: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None,
    base_url: str = "https://api.openai.com/v1",
    seed: int = 42,
    top_p: float = 0.95,
    temperature: float = 0.6,
    max_tokens: int = 4000,
    overwrite: bool = False,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    batch_delay: float = 0.5,
    progress_bar: bool = True,
):
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
    
    output_dir = Path(output_dir) if output_dir else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cache_dir = Path(cache_dir) if cache_dir else default_cache_dir
    cache_file = cache_dir / Path(file_path).stem / f"{pretty_name(model)}_openai_seed{seed}_{start_idx}_{end_idx}.jsonl"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load cached data if exists
    cached_data = {}
    if cache_file.exists() and not overwrite:
        with open(cache_file, 'r') as f:
            for line in f.readlines():
                if line.strip():
                    item = json.loads(line)
                    cached_data[item['id']] = item
        print(f"Loaded {len(cached_data)} cached items from {cache_file}")
    
    # Load data
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
    
    output_file = output_dir / Path(file_path).stem / f"{pretty_name(model)}_openai_seed{seed}_{start_idx}_{end_idx}.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if output_file.exists() and not overwrite:
        print(f"Output file {output_file} already exists. Use --overwrite to overwrite.")
        return
    
    # Identify items that need processing (not in cache)
    items_to_process = []
    final_results = []
    
    for item in data:
        item_id = item["id"]
        if item_id in cached_data:
            # Use cached result
            final_results.append(cached_data[item_id])
        else:
            # Needs processing
            items_to_process.append(item)
            final_results.append(item)  # Will be updated with results later
    
    print(f"Processing {len(data)} items from {start_idx} to {end_idx}...")
    print(f"Found {len(cached_data)} cached items, {len(items_to_process)} items need processing")
    
    if len(items_to_process) == 0:
        print("All items are cached, saving final results...")
        # Save final results
        with open(output_file, 'w') as f:
            for item in final_results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Results saved to {output_file}")
        return
    
    print(f"Model: {model}")
    print(f"Base URL: {base_url}")
    print(f"Temperature: {temperature}")
    print(f"Top-p: {top_p}")
    print(f"Max tokens: {max_tokens}")
    print(f"Seed: {seed}")
    print(f"Batch size: {batch_size}")
    print(f"Max concurrent requests per batch: {max_concurrent}")

    # Preprocess prompts for items that need processing
    messages_list, qids = preprocess_prompts(items_to_process)
    
    # Create mapping for updating final results
    id_to_result_idx = {item["id"]: idx for idx, item in enumerate(final_results)}

    # Create async client context
    start_time = time.time()
    async with OpenAIAsyncClient(api_key=api_key, base_url=base_url) as client:
        # Process in batches
        total_processed = 0
        num_batches = (len(messages_list) + batch_size - 1) // batch_size
        
        batch_iterator = range(0, len(messages_list), batch_size)
        if progress_bar:
            batch_iterator = tqdm(batch_iterator, desc="Processing batches", unit="batch")
        
        for i in batch_iterator:
            batch_messages = messages_list[i:i + batch_size]
            batch_qids = qids[i:i + batch_size]
            batch_data = items_to_process[i:i + batch_size]
            
            if not progress_bar:
                print(f"\nProcessing batch {i//batch_size + 1}/{num_batches}")
            
            # Generate responses for this batch
            batch_start_time = time.time()
            responses = await process_batch(
                client=client,
                messages_batch=batch_messages,
                qids_batch=batch_qids,
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=seed,
                max_retries=max_retries,
                retry_delay=retry_delay,
                max_concurrent=max_concurrent
            )
            batch_time = time.time() - batch_start_time
            
            # Process responses and update data
            batch_results = []
            for j, response in enumerate(responses):
                # Clean up response
                if response and not response.startswith("ERROR:"):
                    response = response.strip()
                    # Remove common end tokens
                    for end_token in ["<|im_end|>", "<|end_of_text|>", "<|eot_id|>"]:
                        if end_token in response:
                            idx = response.index(end_token)
                            response = response[:idx]
                            break
                
                # Create result item
                result_item = batch_data[j].copy()
                result_item['llm_response'] = response
                result_item['qid'] = batch_qids[j]
                result_item['messages'] = batch_messages[j]
                result_item['generation_params'] = {
                    'model': model,
                    'temperature': temperature,
                    'top_p': top_p,
                    'max_tokens': max_tokens,
                    'seed': seed
                }
                
                batch_results.append(result_item)
                
                # Update final results
                result_idx = id_to_result_idx[batch_qids[j]]
                final_results[result_idx] = result_item
            
            # Save batch to cache
            for result_item in batch_results:
                append_jsonl(cache_file, result_item)
            
            total_processed += len(batch_messages)
            if not progress_bar:
                print(f"Completed {total_processed}/{len(messages_list)} items in {batch_time:.1f}s")
                print(f"Throughput: {len(batch_messages)/batch_time:.1f} requests/second")
                print(f"Saved batch {i//batch_size + 1} to cache ({len(batch_results)} items)")
            
            # Add delay between batches to be respectful to the API
            if batch_delay > 0 and i + batch_size < len(messages_list):
                await asyncio.sleep(batch_delay)

    # Save final results
    with open(output_file, 'w') as f:
        for item in final_results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    total_time = time.time() - start_time
    print(f"\nGenerated responses saved to {output_file}")
    print(f"Total processing time: {total_time:.1f}s")
    if len(items_to_process) > 0:
        print(f"Average throughput: {len(items_to_process)/total_time:.1f} requests/second")
    
    # Remove cache file
    if cache_file.exists():
        os.remove(cache_file)
        print(f"Cache file {cache_file} removed")


def main(
    file_path: str,
    output_dir: str = None,
    cache_dir: str = None,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    batch_size: int = 20,
    max_concurrent: int = 10,
    model: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None,
    base_url: str = "https://api.openai.com/v1",
    seed: int = 42,
    top_p: float = 0.95,
    temperature: float = 0.6,
    max_tokens: int = 4000,
    overwrite: bool = False,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    batch_delay: float = 0.5,
    progress_bar: bool = True,
):
    """
    Synchronous wrapper for the async main function.
    """
    start_time = time.time()
    
    try:
        asyncio.run(main_async(
            file_path=file_path,
            output_dir=output_dir,
            cache_dir=cache_dir,
            start_idx=start_idx,
            end_idx=end_idx,
            batch_size=batch_size,
            max_concurrent=max_concurrent,
            model=model,
            api_key=api_key,
            base_url=base_url,
            seed=seed,
            top_p=top_p,
            temperature=temperature,
            max_tokens=max_tokens,
            overwrite=overwrite,
            max_retries=max_retries,
            retry_delay=retry_delay,
            batch_delay=batch_delay,
            progress_bar=progress_bar,
        ))
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    fire.Fire(main)

"""
Usage examples:

# High performance settings with aiohttp
python step2.1_openai_gen.py outputs/step1.1_parsing/Magicoder_Evol_Instruct_110K_gpt_4o_mini.jsonl \
    --start_idx=0 \
    --end_idx=50 \
    --batch_size=25 \
    --max_concurrent=25 \
    --model='gpt-4.1-mini' \
    --top_p=0.95 \
    --temperature=0.6 \
    --max_tokens=4000

# Conservative settings for rate limit sensitive scenarios
python step2.1_openai_gen.py outputs/step1.1_parsing/Magicoder_Evol_Instruct_110K_gpt_4o_mini.jsonl \
    --start_idx=0 \
    --end_idx=100 \
    --batch_size=10 \
    --max_concurrent=5 \
    --model='gpt-4' \
    --batch_delay=2.0

# Using custom OpenAI-compatible API
python step2.1_openai_gen.py outputs/step1.1_parsing/Magicoder_Evol_Instruct_110K_gpt_4o_mini.jsonl \
    --start_idx=0 \
    --end_idx=500 \
    --batch_size=30 \
    --max_concurrent=15 \
    --model='llama-3-70b' \
    --base_url='https://your-custom-endpoint.com/v1' \
    --api_key='your-api-key'

# Maximum throughput (be careful with rate limits!)
python step2.1_openai_gen.py outputs/step1.1_parsing/Magicoder_Evol_Instruct_110K_gpt_4o_mini.jsonl \
    --start_idx=0 \
    --end_idx=2000 \
    --batch_size=100 \
    --max_concurrent=50 \
    --batch_delay=0.1 \
    --retry_delay=0.5

# Resume interrupted job (cached items will be skipped)
python step2.1_openai_gen.py outputs/step1.1_parsing/Magicoder_Evol_Instruct_110K_gpt_4o_mini.jsonl \
    --start_idx=0 \
    --end_idx=1000 \
    --batch_size=25 \
    --max_concurrent=15 \
    --model='gpt-4o-mini'
"""