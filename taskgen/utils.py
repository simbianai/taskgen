import ast
import heapq
import inspect
import re


def get_source_code_for_func(fn):
    if fn.__name__ == "<lambda>":
        source_line = inspect.getsource(fn)
        source_line = source_line.split('#')[0]
        match = re.search(r"\blambda\b[^:]+:.*", source_line)
        if match is None:
            raise ValueError(f"Could not extract lambda function from source: {source_line}")
        match = match.group(0)
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





