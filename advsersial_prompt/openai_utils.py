# import openai
# from tenacity import retry, wait_random_exponential, stop_after_attempt

# # 自动重试，适用于 ChatGPT API 的同步调用
# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
# def generate_with_retry_sync(prompt, model="gpt-4", **kwargs):
#     response = openai.ChatCompletion.create(
#         model=model,
#         messages=[{"role": "user", "content": prompt}],
#         **kwargs
#     )
#     return response["choices"][0]["message"]["content"]

# class OpenAISyncClient:
#     def __init__(self, model="gpt-4"):
#         self.model = model

#     def generate(self, prompt, **kwargs):
#         return generate_with_retry_sync(prompt, model=self.model, **kwargs)

import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt
import os

class OpenAISyncClient:
    def __init__(self, api_key=None, base_url="https://api.openai.com/v1"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate(self, prompt, model="gpt-3.5-turbo", **kwargs):
        messages = [{"role": "user", "content": prompt}]
        return generate_with_retry_sync(self.client, messages, model=model, **kwargs)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_with_retry_sync(openai_client, messages, model="gpt-3.5-turbo", **kwargs):
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs
    )
    return [choice.model_dump() for choice in response.choices] 
