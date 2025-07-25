import json
import os
import time
import hashlib
import regex as re
from typing import Any, Dict, List, Union
def load_jsonl(file_path: str) -> List[Dict[Any, Any]]:
    """load a .jsonl file. Return a List of dictionary, where each dictionary is a line in the file"""
    if not os.path.exists(file_path):
        raise Exception(f"{file_path} Does not exist!!!!")
    with open(file_path, "r") as f:
        lst = f.readlines()
    lst = [json.loads(i) for i in lst]
    return lst


def get_python_code_from_string(input: str) -> str:
    """Basically find code wrapped in ```python ... ``` and return it. If none is found then will return the
    empty string"""
    left_index = input.find("```python")
    if left_index < 0:
        return ""
    input = input[left_index + 9 :]
    right_index = input.find("```")
    if right_index < 0:
        return ""
    input = input[:right_index]
    return input


def parse_incomplete_json(input: str) -> Any:
    """A helper function that will:
    1. try to parse the whole thing as json
    2. try to find json object wrapped in ```json ... ``` and parse it
    3. Try to see if the json is incomplete. if so then try to parse the incomplete json

    This will only work when we are missing ]} at the end, modify if you need it for other
    cases.
    """
    input = input.strip()
    left_idx = input.find("```json")
    if left_idx >= 0:
        input = input[left_idx + 7 :]
    right_idx = input.rfind("```")
    if right_idx >= 0:
        input = input[:right_idx]
    try:
        out = json.loads(input)
        return out
    except:
        pass

    # we now assume that the string is incomplete
    while len(input) > 0:
        try:
            data = json.loads(input + "]}")
            return data
        except json.decoder.JSONDecodeError:
            input = input[:-1]
    # we cannot parse this
    return {"question": None, "tests": None}


def remove_print_statements_from_python_program(input: str) -> str:
    lst = input.splitlines()
    lst = [i for i in lst if not i.strip().startswith("print")]
    return "\n".join(lst)


def print_data(file: str, idx: int = 0):
    data = load_jsonl(file)
    data = [row for row in data if row["id"] == idx][0]
    for key in data:
        print(f"----------------{key}:-------------------")
        if type(data[key]) == list:
            for i in data[key]:
                if type(i) == list:
                    # we omit the original inferences for easier print statements
                    for ii in i:
                        print(ii)
                    break
                else:
                    print(i)
            print(f"Contained {len(data[key])} items-----")
        else:
            print(data[key])

def chunking(lst: List[Any], n: int) -> List[List[Any]]:
    """Split a list into a list of list where each sublist is of size n"""
    if n <= 0:
        raise Exception(f"Are you fucking kidding me with n = {n}?")
    if len(lst) <= n:
        return [lst]
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def load_jsonl(file_path: str) -> List[Dict[Any, Any]]:
    """load a .jsonl file. Return a List of dictionary, where each dictionary is a line in the file"""
    if not os.path.exists(file_path):
        raise Exception(f"{file_path} Does not exist!!!!")
    with open(file_path, "r") as f:
        lst = f.readlines()
    output = [json.loads(i) for i in lst]
    return output


def save_jsonl(file_path: str, content: List[Dict[Any, Any]]) -> None:
    """save a .jsonl file."""
    with open(file_path, "w") as f:
        for i in content:
            f.write(json.dumps(i) + "\n")


def append_jsonl(file_path: str, content: List[Dict[Any, Any]]) -> None:
    """append to a .jsonl file."""
    with open(file_path, "a") as f:
        for i in content:
            f.write(json.dumps(i) + "\n")


class MyTimer:
    """A simple timer class where you initialize it, then just call print_runtime everytime you want to time yourself"""

    def __init__(self) -> None:
        self.start = time.time()

    def print_runtime(self, message: str, reset_timer: bool = True) -> None:
        """Print the runtime, the output will be in the form of f"{message} took ..."

        Parameter:
            message: a string indicating what you have done
            reset_timer: whether to reset timer so that next call to this function will show the time in between print_runtime
        """
        runtime = time.time() - self.start
        minute = int(runtime / 60)
        seconds = runtime % 60
        if minute > 0:
            print(f"{message} took {minute} minutes {seconds} seconds")
        else:
            print(f"{message} took {seconds} seconds")

        if reset_timer:
            self.start = time.time()




def hash_messages(messages: Union[str, List[Dict[str, Any]]]) -> str:
    """
    Hash the messages to get a unique identifier for the conversation.
    If messages is a string, it will be hashed directly.
    If messages is a list of dictionaries in openai format, it will be converted to a string and then hashed.
    
    Args:
        messages: Either a string or a list of message dictionaries in OpenAI format
                 (e.g., [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}])
    
    Returns:
        str: SHA-256 hash of the messages as a hexadecimal string
    """
    if isinstance(messages, str):
        # Hash the string directly
        message_str = messages
    elif isinstance(messages, list):
        # Convert list of dictionaries to JSON string for consistent hashing
        # Sort keys to ensure consistent ordering
        message_str = json.dumps(messages, sort_keys=True, separators=(',', ':'))
    else:
        raise TypeError(f"messages must be str or List[Dict[str, Any]], got {type(messages)}")
    
    # Create SHA-256 hash
    hash_obj = hashlib.sha256(message_str.encode('utf-8'))
    return hash_obj.hexdigest()


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

def complex_pretty_name(name: str) -> str:
    """
    Convert a name to a pretty name of model name/path or dataset name/path to serve as file name.
    
    This function handles common model/dataset naming conventions like:
    - Hugging Face model names (e.g., "microsoft/DialoGPT-medium")
    - File paths (e.g., "/path/to/model/checkpoint.bin")
    - URLs (e.g., "https://example.com/model.tar.gz")
    - Names with version numbers, special characters, etc.
    
    Args:
        name (str): The original model or dataset name/path
        
    Returns:
        str: A cleaned, file-safe name suitable for use as a filename
    """
    if not name or not isinstance(name, str):
        return "unnamed"
    
    # Start with the original name
    pretty = name.strip()
    
    # Remove URL protocols and domains
    pretty = re.sub(r'^https?://', '', pretty)
    pretty = re.sub(r'^[^/]+\.com/', '', pretty)
    pretty = re.sub(r'^[^/]+\.org/', '', pretty)
    
    # Extract filename from path if it's a full path
    if '/' in pretty:
        # Take the last meaningful part (could be filename or directory name)
        parts = [p for p in pretty.split('/') if p.strip()]
        if parts:
            pretty = parts[-1]
            # If it has a file extension, remove it
            if '.' in pretty and not pretty.startswith('.'):
                pretty = os.path.splitext(pretty)[0]
    
    # Handle common model naming patterns
    # Replace organization separators with underscores
    pretty = pretty.replace('/', '_')
    pretty = pretty.replace('\\', '_')
    
    # Replace common separators and special characters
    pretty = re.sub(r'[-\s]+', '_', pretty)  # Replace hyphens and spaces with underscores
    pretty = re.sub(r'[^\w\-_.]', '_', pretty)  # Replace special chars except word chars, hyphens, underscores, dots
    
    # Clean up multiple underscores
    pretty = re.sub(r'_+', '_', pretty)
    
    # Remove leading/trailing underscores and dots
    pretty = pretty.strip('_.')
    
    # Handle empty result
    if not pretty:
        return "unnamed"
    
    # Ensure it doesn't start with a number (some filesystems don't like this)
    if pretty and pretty[0].isdigit():
        pretty = f"model_{pretty}"
    
    # Truncate if too long (keeping it reasonable for most filesystems)
    max_length = 100
    if len(pretty) > max_length:
        pretty = pretty[:max_length].rstrip('_.')
    
    return pretty