import ast
import heapq
import inspect
import re

from openai.types.chat import ChatCompletion
from openai.types import CompletionUsage

ADDITIONAL_DATA_KEY = "additional_data_1983c80c"

def get_source_code_for_func(fn):
    if fn.__name__ == "<lambda>":
        source_line = inspect.getsource(fn)
        source_line = source_line.split('#')[0]
        match = re.search(r"\blambda\b[^:]+:.*", source_line).group(0)
        splits = [s for s in match.split(",") if s != ""]
        fn_code = splits[0]
        idx = 1
        while idx < len(splits):
            try:
                ast.parse(fn_code)
                break
            except SyntaxError as _:
                fn_code = fn_code + "," + splits[idx]
                idx = idx + 1
        while True:
            try:
                ast.parse(fn_code)
                break
            except SyntaxError as _:
                fn_code = fn_code[:-1]

        return fn_code
    else:
        return inspect.getsource(fn)
    
    
def ensure_awaitable(func, name):
    """ Utility function to check if the function is an awaitable coroutine function """
    if func is not None and not inspect.iscoroutinefunction(func):
        raise TypeError(f"{name} must be an awaitable coroutine function")
    
    
### Helper Functions
def top_k_index(lst, k):
    ''' Given a list lst, find the top k indices corresponding to the top k values '''
    indexed_lst = list(enumerate(lst))
    top_k_values_with_indices = heapq.nlargest(k, indexed_lst, key=lambda x: x[1])
    top_k_indices = [index for index, _ in top_k_values_with_indices]
    return top_k_indices

def extract_usage_obj(llm_response, host: str) -> dict:
    ''' Extract the usage object from the response '''
    usage_obj = None
    
    if host == "openai":
        if not isinstance(llm_response, ChatCompletion):
            return {"error": "LLM response is not an instance of CompletionUsage class"}
        usage_obj = llm_response.usage

        if not isinstance(usage_obj, CompletionUsage):
            return {"error": "Usage object is not an instance of CompletionUsage class"}
        usage_obj = usage_obj.model_dump()
        usage_obj["model_name"] = llm_response.model
    else:
        usage_obj = {"error": "Unknown host"}

    return usage_obj

def add_additional_data_to_response(llm_responses: dict, response_dict: dict) -> dict:
    ''' Add additional data to the LLM response '''

    for id, llm_response_dict in llm_responses.items():
        host = llm_response_dict.get("host", "openai")
        llm_response = llm_response_dict.get("response", None)

        if not isinstance(response_dict, dict):
            print("response_dict is not a dictionary", response_dict, type(response_dict))
            return response_dict
        
        if ADDITIONAL_DATA_KEY not in response_dict:
            response_dict[ADDITIONAL_DATA_KEY] = {}
        
        # Add Usage object to the response
        usage_obj = extract_usage_obj(llm_response, host)
        if "usage" not in response_dict[ADDITIONAL_DATA_KEY]:
            response_dict[ADDITIONAL_DATA_KEY]["usage"] = []
        
        if usage_obj:   
            response_dict[ADDITIONAL_DATA_KEY]["usage"].append(usage_obj)

    return response_dict