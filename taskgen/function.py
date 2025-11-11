import json
import re
import inspect
from typing import Any, cast, get_type_hints

from taskgen.base import strict_json
from taskgen.base_async import strict_json_async

from taskgen.utils import ensure_awaitable, get_source_code_for_func

### Helper Functions ###

def get_clean_typename(typ) -> str:
    """Returns a clean, readable name for a type, including handling generics."""
    if hasattr(typ, '__origin__'):  # Check for generic types
        if typ.__origin__ is not None:  # Generic types, e.g., List, Dict
            base_name = typ.__origin__.__name__
            if hasattr(typ, '__args__') and typ.__args__ is not None:
                args = [get_clean_typename(arg) for arg in typ.__args__]
                return f"{base_name}[{', '.join(args)}]"
            else:
                return base_name  # Handle cases like `Dict` without specified parameters
        else:  # Non-generic but special types, e.g., typing.List without parameters
            return typ._name if hasattr(typ, '_name') else str(typ)
    elif hasattr(typ, '__name__'):
        return typ.__name__  # Simple types, e.g., int, str
    else:
        return str(typ)  # Fallback, should rarely be used

def get_fn_description(my_function):
    ''' Returns the modified docstring of my_function, that takes into account input variable names and types in angle brackets
    Also returns the list of input parameters to the function in sequence
    e.g.: Adds numbers x and y -> Adds numbers <x: int> and <y: int>
    Input variables that are optional (already assigned a default value) need not be in the docstring
    args and kwargs variables are not parsed '''
     
    if not inspect.isfunction(my_function):
        raise Exception(f'{my_function} is not a Python function')
        
    # Get the signature and type hints of the function
    # if my_function.__doc__ == None:
    #     return '', []

    signature = inspect.signature(my_function)
    full_type_hints = get_type_hints(my_function)
    my_fn_description = my_function.__doc__ if my_function.__doc__ else ''

    param_list = []
    # Access parameters and their types
    parameters = signature.parameters
    for param_name, param in parameters.items():
        # skip args and kwargs and shared variables
        if param_name in ['shared_variables', 'args', 'kwargs']:
            continue
        
        param_type = param.annotation.__name__ if param.annotation != inspect.Parameter.empty else "unannotated"
        # Handle specific typing
        if param_name in full_type_hints:
            param_type = get_clean_typename(full_type_hints[param_name])

        # Create new_param representation
        new_param = f'<{param_name}: {param_type}>' if param_type != "unannotated" else f'<{param_name}>'

        # Pattern to find the parameter in the docstring
        pattern = re.compile(fr'\b({param_name})\b')
        
        # Substitute the parameter in the docstring
        if pattern.search(my_fn_description):
            my_fn_description = pattern.sub(new_param, my_fn_description)
            param_list.append(param_name)
            
        # otherwise, function description will just be the function signature
        else:
            # add a continuation if description is current empty
            if my_fn_description != '':
                my_fn_description += ', '
                
            param_list.append(param_name)
            print(f'Input variable "{param_name}" not in docstring of "{my_function.__name__}". Adding it to docstring')
            my_fn_description += f'Input: {new_param}'
            
            if param.default != inspect.Parameter.empty:
                my_fn_description += f', default: {param.default}'

    return my_fn_description, param_list

def get_fn_output(my_function) -> dict:
    ''' Returns the dictionary of output parameters and types of the form {"Output 1": "Type", "Output 2": "Type"}'''
     
    if not inspect.isfunction(my_function):
        raise Exception(f'{my_function} is not a Python function')
        
    # Initialize the output format dictionary
    output_format = {}

    full_type_hints = get_type_hints(my_function)
    my_fn_description = my_function.__doc__

    # Check for return annotation
    if 'return' in full_type_hints:
        return_type = full_type_hints['return']
        # Adjust dictionary according to the return type
        if isinstance(return_type, tuple):
            for idx, type_hint in enumerate(return_type):
                output_format[f"output_{idx + 1}"] = get_clean_typename(type_hint)
        else:
            output_format["output_1"] = get_clean_typename(return_type)

    return output_format

### Main Class ###

class BaseFunction:
    def __init__(self,
                 fn_description: str = '', 
                 output_format: dict|None = None,
                 examples = None,
                 external_fn = None,
                 is_compulsory = False,
                 fn_name = None,
                 llm = None,
                 **kwargs):
        ''' 
        Creates an LLM-based function or wraps an external function using fn_description and outputs JSON based on output_format. 
        (Optional) Can define the function based on examples (list of Dict containing input and output variables for each example)
        (Optional) If you would like greater specificity in your function's input, you can describe the variable after the : in the input variable name, e.g. `<var1: an integer from 10 to 30`. Here, `var1` is the input variable and `an integer from 10 to 30` is the description.
        
        Inputs (primary):
        - fn_description: String. Function description to describe process of transforming input variables to output variables. Variables must be enclosed in <> and listed in order of appearance in function input.
Can also be done automatically by providing docstring with input variable names in external_fn
        - output_format: Dict. Dictionary containing output variables names and description for each variable.
           
        Inputs (optional):
        - examples: Dict or List[Dict]. Examples in Dictionary form with the input and output variables (list if more than one)
        - external_fn: Python Function. If defined, instead of using LLM to process the function, we will run the external function. 
            If there are multiple outputs of this function, we will map it to the keys of `output_format` in a one-to-one fashion
        - is_compulsory: Bool. Default: False. This is whether to always use the Function when doing planning in Agents
        - fn_name: String. If provided, this will be the name of the function. Otherwise, if `external_fn` is provided, it will be the name of `external_fn`. Otherwise, we will use LLM to generate a function name from the `fn_description`
        - llm: Function. The llm parameter to pass into strict_json
        - **kwargs: Dict. Additional arguments you would like to pass on to the strict_json function (such as llm)
        
        ## Example
        fn_description = 'Output the sum of <num1> and <num2>'
        output_format = {'output': 'sum of two numbers'}
        examples = [{'num1': 5, 'num2': 6, 'output': 11}, {'num1': 2, 'num2': 4, 'output': 6}]
        '''
        
        self.fn_description = ''
        self.output_format = {}
        
        # this is only for external functions
        self.external_param_list = [] 
        if external_fn is not None:
            # add in code to warn user if type is defined for external function
            type_check = False
            if output_format is not None:
                for value in output_format.values():
                    if 'type:' in str(value):
                        type_check = True
                if type_check:
                    print('Note: Type checking (type:) not done for External Functions')
            
            # get details from docstring of external function only if fn_description is not given
            if fn_description == '':
                self.fn_description, self.external_param_list = get_fn_description(external_fn)
            
            # get the output format from the function signature if output format is not given
            if output_format is None:
                self.output_format = get_fn_output(external_fn)             
            
        # if function description provided, use it to update the function description
        if fn_description != '':
            self.fn_description = fn_description

        # if output format is provided, use it to update the function output format
        if output_format is not None:
            self.output_format = output_format
            
        self.examples = examples
        self.external_fn = external_fn
        self.is_compulsory = is_compulsory
        self.fn_name = fn_name
        self.llm = llm
        self.kwargs = kwargs
        
        self.variable_names = []
        self.shared_variable_names = []
        # use regex to extract variables from function description
        matches = re.findall(r'<(.*?)>', self.fn_description)
            
        for match in matches:
            first_half = match.split(':')[0]
            if first_half not in self.variable_names:
                # if the first two characters of variable are s_, means take from shared_variables
                if first_half[:2] != 's_':
                    self.variable_names.append(first_half)
                # otherwise we take from shared_variables
                else:
                    self.shared_variable_names.append(first_half)
                    # replace the original variable without the <> so as not to confuse the LLM
                    self.fn_description = self.fn_description.replace(f'<{match}>', first_half)
                    
        # make it such that we follow the same order for variable names as per the external function only if there are external function params
        if self.external_param_list != []:
            self.variable_names = [x for x in self.external_param_list if x in self.variable_names]
             

        # Append examples to description
        if self.examples is not None:
            self.fn_description += '\nExamples:\n' + str(examples) 
            
    def __str__(self):
        ''' Prints out the function's parameters '''
        return f'Description: {self.fn_description}\nInput: {self.variable_names}\nOutput: {self.output_format}\n'
    
    def get_python_representation(self):
        """Returns a Python representation of the Function object, including the external function code if available."""
        external_fn_code = None
        external_fn_ref = None

        if self.external_fn:
            if inspect.isfunction(self.external_fn) and self.external_fn.__name__ == "<lambda>":
                external_fn_ref = get_source_code_for_func(self.external_fn)
            else:
                external_fn_ref = self.external_fn.__name__
                external_fn_code = get_source_code_for_func(self.external_fn)

        fn_initialization = f"""Function(
            fn_name="{self.fn_name}",
            fn_description='''{self.fn_description}''',
            output_format={self.output_format},
            examples={self.examples},
            external_fn={external_fn_ref},
            is_compulsory={self.is_compulsory})
        """
        return (fn_initialization, external_fn_code)
    
    
    def _prepare_function_kwargs(self, *args, **kwargs):
         # get the shared_variables if there are any
        shared_variables = kwargs.get('shared_variables', {})
        # remove the mention of shared_variables in kwargs
        if 'shared_variables' in kwargs:
            del kwargs['shared_variables']
        # extract out only variables listed in variable_list from kwargs
        function_kwargs = {key: value for key, value in kwargs.items() if key in self.variable_names}
        # additionally, if function references something in shared_variables, add that in
        for variable in self.shared_variable_names:
            if variable in shared_variables:
                function_kwargs[variable] = shared_variables[variable]
        # Do the auto-naming of variables as var1, var2, or as variable names defined in variable_names
        for num, arg in enumerate(args):
            if len(self.variable_names) > num:
                function_kwargs[self.variable_names[num]] = arg
            else:
                function_kwargs[f'var{num+1}'] = arg
                
        return function_kwargs, shared_variables

    def _prepare_strict_json_kwargs(self, **kwargs):
        return {key: value for key, value in kwargs.items() if key not in self.variable_names}

    def _update_shared_variables(self, results, shared_variables):
        keys_to_delete = []
        for key in results:
            if key.startswith('s_'):
                shared_variables[key] = results[key]
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del results[key]

    def _generate_function_name(self):
        if self.fn_name is None:
            if self.external_fn is not None and hasattr(self.external_fn, '__name__') and self.external_fn.__name__ != '<lambda>':
                self.fn_name = self.external_fn.__name__
            else:
                self.fn_name = 'generated_function_name'  # Replace with actual function name generation logic.
            self.__name__ = self.fn_name
    
    
            
            
class Function(BaseFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.fn_name is None:
            # if external function has a name, use it
            if self.external_fn is not None and hasattr(self.external_fn, '__name__') and self.external_fn.__name__ != '<lambda>':
                self.fn_name = self.external_fn.__name__
            # otherwise, generate name out
            else:
                res = cast(dict[str, Any], strict_json(system_prompt = "Output a function name to summarise the usage of this function.",
                                  user_prompt = str(self.fn_description),
                                  output_format = {"Thoughts": "What function does", "Name": "Function name with _ separating words that summarises what function does"},
                                    llm = self.llm,
                                    **self.kwargs))
                self.fn_name = res['Name']
         # change instance's name to function's name
        self.__name__ = self.fn_name
        
    
    def __call__(self, *args, **kwargs):
        ''' Describes the function, and inputs the relevant parameters as either unnamed variables (args) or named variables (kwargs)
        
        Inputs:
        - shared_varables: Dict. Default: empty dict. The variables which will be shared between functions. Only passed in if required by function 
        - *args: Tuple. Unnamed input variables of the function. Will be processed to var1, var2 and so on based on order in the tuple
        - **kwargs: Dict. Named input variables of the function. Can also be variables to pass into strict_json
        
        Output:
        - res: Dict. JSON containing the output variables'''
        
        # get the shared_variables if there are any
        function_kwargs, shared_variables = self._prepare_function_kwargs(*args, **kwargs)

        # extract out only variables not listed in variable list
        strict_json_kwargs = {
            my_key: kwargs[my_key] for my_key in kwargs 
            if my_key not in self.variable_names and my_key != 'shared_variables'
        }
                
        # If strict_json function, do the function. 
        if self.external_fn is None:
            res = strict_json(system_prompt = self.fn_description,
                            user_prompt = json.dumps(function_kwargs),
                            output_format = self.output_format,
                            llm = self.llm,
                            **self.kwargs, **strict_json_kwargs)
            
        # Else run the external function
        else:
            res = {}
            # if external function uses shared_variables, pass it in
            argspec = inspect.getfullargspec(self.external_fn)
            if 'shared_variables' in argspec.args:
                fn_output = self.external_fn(shared_variables = shared_variables, **function_kwargs)
            else:
                fn_output = self.external_fn(**function_kwargs)
                
            # if there is nothing in fn_output, skip this part
            if fn_output is not None:
                output_keys = list(self.output_format.keys())
                # convert the external function into a tuple format to parse it through the JSON dictionary output format
                if not isinstance(fn_output, tuple):
                    fn_output = [fn_output]

                for i in range(len(fn_output)):
                    if len(output_keys) > i:
                        res[output_keys[i]] = fn_output[i]
                    else:
                        res[f'output_{i+1}'] = fn_output[i]
        
        # check if any of the output variables have a s_, which means we update the shared_variables and not output it
        self._update_shared_variables(res, shared_variables)
                
        if res == {}:
            res = {'Status': 'Completed'}

        return res
        
        
            
class AsyncFunction(BaseFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ensure_awaitable(self.llm, 'llm')
        if self.fn_name is None:
            # if external function has a name, use it
            if self.external_fn is not None and hasattr(self.external_fn, '__name__') and self.external_fn.__name__ != '<lambda>':
                self.fn_name = self.external_fn.__name__
        self.__name__ = self.fn_name
        
    async def async_init(self): 
        ''' This generates the name for the function using strict_json_async '''
        if self.fn_name is None:
            res = cast(dict[str, Any], await strict_json_async(system_prompt = "Output a function name to summarise the usage of this function.",
                              user_prompt = str(self.fn_description),
                              output_format = {"Thoughts": "What function does", "Name": "Function name with _ separating words that summarises what function does"},
                             llm = self.llm,
                             **self.kwargs))
            self.fn_name = res['Name']

            # change instance's name to function's name
            self.__name__ = self.fn_name
        
    async def __call__(self, *args, **kwargs):
        ''' Describes the function, and inputs the relevant parameters as either unnamed variables (args) or named variables (kwargs)
        
        Inputs:
        - shared_varables: Dict. Default: empty dict. The variables which will be shared between functions. Only passed in if required by function 
        - *args: Tuple. Unnamed input variables of the function. Will be processed to var1, var2 and so on based on order in the tuple
        - **kwargs: Dict. Named input variables of the function. Can also be variables to pass into strict_json
        
        Output:
        - res: Dict. JSON containing the output variables'''
        
        # get the shared_variables if there are any
      
        if self.fn_name is None:
            await self.async_init()
        
        function_kwargs, shared_variables = self._prepare_function_kwargs(*args, **kwargs)

        # extract out only variables not listed in variable list
        strict_json_kwargs = {
                    my_key: kwargs[my_key] for my_key in kwargs 
                    if my_key not in self.variable_names and my_key != 'shared_variables'
                }
               
                
        # If strict_json function, do the function. 
        if self.external_fn is None:
            res = cast(dict[str, Any], await strict_json_async(system_prompt = self.fn_description,
                            user_prompt = json.dumps(function_kwargs),
                            output_format = self.output_format,
                            llm = self.llm,
                            **self.kwargs, **strict_json_kwargs))
            
        # Else run the external function
        else:
            res = {}
            # if external function uses shared_variables, pass it in
            argspec = inspect.getfullargspec(self.external_fn)
            if 'shared_variables' in argspec.args:
                if  inspect.iscoroutinefunction(self.external_fn):
                    fn_output = await self.external_fn(shared_variables = shared_variables, **function_kwargs)
                else: 
                    fn_output = self.external_fn(shared_variables = shared_variables, **function_kwargs)
            else:
                if  inspect.iscoroutinefunction(self.external_fn):
                    fn_output = await self.external_fn(**function_kwargs)
                else:
                    fn_output = self.external_fn(**function_kwargs)
                
            # if there is nothing in fn_output, skip this part
            if fn_output is not None:
                output_keys = list(self.output_format.keys())
                # convert the external function into a tuple format to parse it through the JSON dictionary output format
                if not isinstance(fn_output, tuple):
                    fn_output = [fn_output]

                for i in range(len(fn_output)):
                    if len(output_keys) > i:
                        res[output_keys[i]] = fn_output[i]
                    else:
                        res[f'output_{i+1}'] = fn_output[i]
        
         
        # check if any of the output variables have a s_, which means we update the shared_variables and not output it
        self._update_shared_variables(res, shared_variables)
                
        if res == {}:
            res = {'Status': 'Completed'}

        return res


    
# alternative name for strict_function (it is now called Function)
strict_function = Function