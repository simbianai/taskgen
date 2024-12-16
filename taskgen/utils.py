import ast
import heapq
import inspect
import re

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

def extract_usage_obj(llm_response: dict, host: str="openai") -> dict:
    ''' Extract the usage object from the response '''
    if not isinstance(llm_response, dict):
        return {"error": "LLM response is not a dictionary"}
    
    usage_obj = None
    if host == "openai":
        usage_obj = llm_response.get("usage")

        if usage_obj is None or not isinstance(usage_obj, dict):
            return {"error": "Usage object not found in the response or is not a dictionary"}
        usage_obj = usage_obj.model_dump()

    return usage_obj

def add_additional_data_to_response(llm_response, response_dict: dict, host: str="openai") -> dict:
    ''' Add additional data to the LLM response '''
    if not isinstance(response_dict, dict):
        return response_dict
    
    if ADDITIONAL_DATA_KEY not in response_dict:
        response_dict[ADDITIONAL_DATA_KEY] = {}
    
    # Add Usage object to the response
    usage_obj = extract_usage_obj(llm_response, host)
    if "usage" not in response_dict[ADDITIONAL_DATA_KEY]:
        response_dict[ADDITIONAL_DATA_KEY]["usage"] = []

    response_dict[ADDITIONAL_DATA_KEY]["usage"].append(usage_obj)

    return response_dict