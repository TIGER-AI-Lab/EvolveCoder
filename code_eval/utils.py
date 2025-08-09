import hashlib
import ast
import regex as re
from .prime_code import compute_score as prime_code_compute_score
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from code_eval.acecoder import evaluate_test_cases

def hash_string(s):
    return hashlib.sha256(s.encode()).hexdigest()

def check_syntax(code_string):
    try:
        # Attempt to parse the code string
        ast.parse(code_string)
        return True
    # except SyntaxError as e:
    except Exception as e:
        # If a SyntaxError is raised, the code is not valid
        # print(f"Syntax error in code: {e}")
        return False
    
def parse_code(action: str, mode="all"):
    """
    Parse the raw action string (which is the llm response) into an actual action and its contents.
    Ensures that the parsed code is valid and safe for execution.
    
    Args:
        action: Raw action string containing Python code
        
    Returns:
        Tuple containing the extracted code and a validity flag
    """
    # Try to find Python code in various formats
    all_valid_python_code = re.findall(r"<python>(.*?)</python>", action, re.DOTALL)
    
    if not all_valid_python_code:
        all_valid_python_code = re.findall(r"```\n?python(.*?)```", action, re.DOTALL)
    
    if len(all_valid_python_code) == 0:
        return ""
    
    if mode == "all":
        parsed_code = "\n".join([code for code in all_valid_python_code if check_syntax(code)])
    elif mode == "first":
        # Use the first code block found
        parsed_code = all_valid_python_code[0]
    elif mode == "last":
        # Use the last code block found
        parsed_code = all_valid_python_code[-1]
    elif mode == "all_in_last_turn":
        # parse all the code blocks only in the last assistant turn
        # find the last assistant turn
        last_turn_start_idx = action.rfind('<|im_start|>assistant')
        if last_turn_start_idx == -1:
            last_turn = action
        else:
            last_turn = action[last_turn_start_idx:]
        all_valid_python_code = re.findall(r"<python>(.*?)</python>", last_turn, re.DOTALL)
        if not all_valid_python_code:
            all_valid_python_code = re.findall(r"```\n?python(.*?)```", last_turn, re.DOTALL)
        if len(all_valid_python_code) == 0:
            return ""
        parsed_code = "\n".join([code for code in all_valid_python_code if check_syntax(code)])
    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'all', 'first', 'last', or 'all_in_last_turn'.")
    
    parsed_code = parsed_code.strip(' \n')
    return parsed_code