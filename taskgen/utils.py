import ast
import heapq
import inspect
import re

from openai.types.chat import ChatCompletion
from openai.types import CompletionUsage


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

def extract_usage_from_responses(llm_responses: dict) -> list:
    ''' Extract the usage object from the LLM response '''
    usage_objects = []
    for id, llm_response_dict in llm_responses.items():
        host = llm_response_dict.get("host", "openai")
        llm_response = llm_response_dict.get("response", None)

        if host == "openai" or host == "azure-openai":
            if not isinstance(llm_response, ChatCompletion):
                usage_objects.append({"error": "LLM response is not an instance of CompletionUsage class"})
                continue
            
            usage_object = llm_response.usage

            if not isinstance(usage_object, CompletionUsage):
                usage_objects.append({"error": "Usage object is not an instance of CompletionUsage class"})
                continue
            usage_object = usage_object.model_dump()
            usage_object["model_name"] = llm_response.model
            usage_object["host"] = host
            usage_objects.append(usage_object)
        else:
            usage_objects.append({"error": "Unknown host"})

    return usage_objects

def get_final_response(raw_llm_responses: dict, response_dict: dict, include_additional_data: False) -> dict:
    ''' Extract the final response from the LLM response '''

    if not include_additional_data:
        return response_dict
    
    usage = extract_usage_from_responses(raw_llm_responses)
    return {
        "response_dict": response_dict,
        "usage": usage,
        "raw_llm_responses": raw_llm_responses,
    }