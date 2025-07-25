
import asyncio
import aiohttp
from typing import Optional, List, Dict

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
        n: int = 1,
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
            "n": n,
            "max_tokens": max_tokens,
        }
        
        if seed is not None:
            payload["seed"] = seed
        
        async with self.session.post(url, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                return [data["choices"][i]["message"]["content"] for i in range(len(data["choices"]))]
            else:
                error_text = await response.text()
                raise aiohttp.ClientResponseError(
                    request_info=response.request_info,
                    history=response.history,
                    status=response.status,
                    message=f"OpenAI API error: {error_text}"
                )

async def generate_with_retry(
    client: OpenAIAsyncClient,
    messages: List[dict],
    model: str,
    temperature: float,
    top_p: float,
    n: int,
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
                    n=n,
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
