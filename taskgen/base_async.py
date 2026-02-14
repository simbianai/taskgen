import asyncio
import json
import re
import ast
from typing import Tuple
from taskgen.base import convert_to_dict, parse_response_llm_check, type_check_and_convert, wrap_with_angle_brackets

from taskgen.utils import ensure_awaitable

### Helper Functions ###


async def convert_to_list_async(field: str, **kwargs) -> list:
    '''Converts the string field into a list using the LLM (with **kwargs) to list out elements line by line'''
    
    system_msg = '''Output each element of the list in a new line starting with (%item) and ending with \n, e.g. ['hello', 'world'] -> (%item) hello\n(%item) world\nStart your response with (%item) and do not provide explanation'''
    user_msg = str(field)
    res = await chat_async(system_msg, user_msg, **kwargs)

    # Extract out list items
    items = re.findall(r'\(%item\)\s*(.*?)\n*(?=\(%item\)|$)', res, flags=re.DOTALL)
    return items



async def llm_check_async(field, llm_check_msg: str, **kwargs) -> Tuple[bool, str]:
    ''' Uses the LLM to check if the field adheres to the llm_check_msg.
    Outputs whether requirement is met (True or False) and the action needed'''
    system_msg = f'''Check whether output field meets this requirement: {llm_check_msg}
Output in the following format:
```
# Thoughts: <Thoughts about whether output field meets requirement>
# Requirement Met: <Yes or No>
# Action Needed: <If Requirement Met is No, state in one sentence how to meet requirement. Otherwise, output NA>"
```
Update text enclosed in <>. Be concise.
'''
    user_msg = str(field)
    res = await chat_async(system_msg, user_msg, **kwargs)
    requirement_met, action_needed = parse_response_llm_check(res)
    return requirement_met, action_needed


async def check_datatype_async(field, key, data_type: str, **kwargs):
    ''' Ensures that output field of the key of JSON dictionary is of data_type 
    Currently supports int, float, str, code, enum, lists, nested lists, dict, dict with keys
    Takes in **kwargs for the LLM model
    Returns corrected output field that matches the datatype'''
    data_type = data_type.strip()
    
    # check if we want an LLM-based correction
    if data_type.lower()[:6] == 'ensure':
        llm_check_msg = data_type[6:].strip()
        print(f'Using LLM to check "{field}" to see if it adheres to "{llm_check_msg}"')
        requirement_met, action_needed = await llm_check_async(field, llm_check_msg, **kwargs)
        # if check failed, raise error
        if not requirement_met:
            raise Exception(f'''Output field of "{key}" does not meet requirement "{llm_check_msg}". Action needed: "{action_needed}"''')
            
    # check for list at beginning of datatype
    # or the output field begins with [ and ends with ] but it is not a list, indicating an error with ast.literal_eval
    if data_type.lower()[:4] == 'list' or data_type.lower()[:5] == 'array' or (str(field)[0]=='[' and str(field)[-1]==']'):
        # first try to see if we can do ast.literal_eval with { and }
        try:
            field = str(field)
            startindex = field.find('[')
            endindex = field.rfind(']')
            field = field[startindex: endindex+1]
            field = ast.literal_eval(field)
        except Exception as e:
            pass
        if not isinstance(field, list):
            # if it is already in a datatype that is a list, ask LLM to fix it (1 LLM call)
            if '[' in field and ']' in field:
                print(f'Attempting to use LLM to fix {field} as it is not a proper array')
                field = await convert_to_list_async(field, **kwargs)   
                print(f'Fixed list: {field}\n\n')
            else:
                raise Exception(f'''Output field of "{key}" not of data type array. If not possible to match, split output field into parts for elements of the array''')
            
    # check for nested list
    # Regex pattern to match content inside square brackets
    match = re.search(r"list\[(.*)\]", data_type, re.IGNORECASE)
    if match:
        internal_data_type = match.group(1)  # Extract the content inside the brackets
        # do processing for internal elements
        for num in range(len(field)):
            field[num] = await check_datatype_async(field[num], 'array element of '+key, internal_data_type, **kwargs)
            
    match = re.search(r"array\[(.*)\]", data_type, re.IGNORECASE)
    if match:
        internal_data_type = match.group(1)  # Extract the content inside the brackets
        # do processing for internal elements
        for num in range(len(field)):
            field[num] = await check_datatype_async(field[num], 'array element of '+key, internal_data_type, **kwargs)
            
    # if it is not nested, check individually
    else:
        field = type_check_and_convert(field, key, data_type, **kwargs)
    return field



    
    
    
async def check_key_async(field, output_format, new_output_format, delimiter: str, delimiter_num: int, **kwargs):
    ''' Check whether each key in dict, or elements in list of new_output_format is present in field, and whether they meet the right data type requirements, then convert field to the right data type
    If needed, calls LLM model with parameters **kwargs to correct the output format for improperly formatted list
    output_format is user-given output format at each level, new_output_format is with delimiters in keys, and angle brackets surrounding values
    If output_format is a string, decode escape characters, so that code can run
    Returns field that is converted to a dictionary if able to. Otherwise, raises Exception errors for missing keys or wrong output format'''
    
    cur_delimiter = delimiter*delimiter_num
    
    if isinstance(output_format, dict):   
        # this is the processed output dictionary for that particular layer in the output structure
        output_d = {}
        # check key appears for each element in the output
        output_d = convert_to_dict(field, list(output_format.keys()), cur_delimiter)
            
        # after creating dictionary, step into next layer
        for key, value in output_d.items():
            # # if the output is a bool type, convert true and false into True and False for ast.literal_eval parsing
            if isinstance(output_format[key], str) and 'type:' in output_format[key] and 'bool' in output_format[key].split('type:')[-1]:
                value = value.replace('true','True').replace('false','False')
            output_d[key] = await check_key_async(value, output_format[key], new_output_format[cur_delimiter+key+cur_delimiter], delimiter, delimiter_num+1)
            # after stepping back from the later layers back to present layer, check for types
            if isinstance(output_format[key], str) and 'type:' in output_format[key]:             
                # extract out data type
                data_type = str(output_format[key]).split('type:')[-1]
                # check the data type, perform type conversion as necessary
                output_d[key] = await check_datatype_async(output_d[key], key, data_type, **kwargs)   
                
        return output_d

    # if list, step into each element
    elif isinstance(output_format, list):
        try:
            field = ast.literal_eval(field)
        except Exception as e:
            # if there is an error in literal processing, use LLM to split field into list
            field = await convert_to_list_async(field, **kwargs)
            
        # check that list has at least same number of elements as the input
        if len(field) < len(output_format):
            raise Exception(f'''Output "{field}" has fewer elements than required by "{output_format}". Add in more list elements.''')
        
        coroutines = [check_key_async(str(field[num]), output_format[num], new_output_format[num], delimiter, delimiter_num+1) for num in range(len(output_format))]
        results = await asyncio.gather(*coroutines)
        return results
    
    # if string, then do literal eval to convert output field for further processing
    elif isinstance(output_format, str):
        # if literal eval fails, just leave it as string, no need to raise error
        try:
            field = ast.literal_eval(field)
        except Exception as e:
            pass
        return field
    
    # otherwise just return the value
    else:
        return field
    


    



async def chat_async(system_prompt: str, user_prompt: str|list[str|dict], model: str = 'gpt-4o-mini', temperature: float = 0, verbose: bool = False, host: str = 'openai', llm= None, **kwargs):
    r"""Performs a chat with the host's LLM model with system prompt, user prompt, model, verbose and kwargs
    Returns the output string res
    - system_prompt: String. Write in whatever you want the LLM to become. e.g. "You are a \<purpose in life\>"
    - user_prompt: String or list of strings. The user input. Later, when we use it as a function, this is the function input
    - model: String. The LLM model to use for json generation
    - verbose: Boolean (default: False). Whether or not to print out the system prompt, user prompt, GPT response
    - host: String. The provider of the LLM
    - llm: User-made llm function.
        - Inputs:
            - system_prompt: String. Write in whatever you want the LLM to become. e.g. "You are a \<purpose in life\>"
            - user_prompt: String. The user input. Later, when we use it as a function, this is the function input
        - Output:
            - res: String. The response of the LLM call
    - **kwargs: Dict. Additional arguments for LLM chat
    """
    res = ""
    if llm is not None:
        ensure_awaitable(llm, 'llm')
        ''' If you specified your own LLM, then we just feed in the system and user prompt 
        LLM function should take in system prompt (str) and user prompt (str), and output a response (str) '''
        res = await llm(system_prompt = system_prompt, user_prompt = user_prompt)
    
    ## This part here is for llms that are OpenAI based
    elif host == 'openai':
        # additional checks for openai json mode
        if 'response_format' in kwargs and kwargs['response_format'] == {"type": "json_object"}:
            # if model fails, default to gpt-3.5-turbo-1106
            try:
                assert(model in ['gpt-4-1106-preview', 'gpt-3.5-turbo-1106'])
            except Exception as e:
                model = 'gpt-3.5-turbo-1106'

        from openai import AsyncOpenAI
        client = AsyncOpenAI()
        if isinstance(user_prompt, list):
            user_prompt = "\n".join([prompt.get("text", "") if isinstance(prompt, dict) and prompt.get("text", "") else str(prompt) for prompt in user_prompt])
        response = await client.chat.completions.create(
            model=model,
            temperature = temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            **kwargs
        )
        res = response.choices[0].message.content

    if verbose:
        print('System prompt:', system_prompt)
        print('\nUser prompt:', user_prompt)
        print('\nGPT response:', res)
            
    return res



### Main Functions ###
    
    
async def strict_json_async(system_prompt: str, user_prompt: str|list[str|dict], output_format: dict, return_as_json = False, custom_checks: dict|None = None, check_data = None, delimiter: str = '###', num_tries: int = 3, openai_json_mode: bool = False, **kwargs):
    r""" Ensures that OpenAI will always adhere to the desired output JSON format defined in output_format.
    Uses rule-based iterative feedback to ask GPT to self-correct.
    Keeps trying up to num_tries it it does not. Returns empty JSON if unable to after num_tries iterations.
    
    Inputs (compulsory):
    - system_prompt: String. Write in whatever you want GPT to become. e.g. "You are a \<purpose in life\>"
    - user_prompt: String or list of strings or dictionaries. The user input. Later, when we use it as a function, this is the function input
    - output_format: Dict. JSON format with the key as the output key, and the value as the output description
    
    Inputs (optional):
    - return_as_json: Bool. Default: False. Whether to return the output as a json. If False, returns as Python dict. If True, returns as json string
    - custom_checks: Dict. Key is output key, value is function which does checking of content for output field
    - check_data: Any data type. The additional data for custom_checks to use if required
    - delimiter: String (Default: '###'). This is the delimiter to surround the keys. With delimiter ###, key becomes ###key###
    - num_tries: Integer (default: 3). The number of tries to iteratively prompt GPT to generate correct json format
    - openai_json_mode: Boolean (default: False). Whether or not to use OpenAI JSON Mode
    - **kwargs: Dict. Additional arguments for LLM chat
    
    Output:
    - res: Dict. The JSON output of the model. Returns {} if JSON parsing failed.
    """
    # default initialise custom_checks to {}
    if custom_checks is None:
        custom_checks = {}
        
    # If OpenAI JSON mode is selected, then just let OpenAI do the processing
    if openai_json_mode:
        # add in code to warn user if type is defined for external function
        type_check = False
        for value in output_format.values():
            if 'type:' in str(value):
                type_check = True
        if type_check:
            print('Note: Type checking (type:) not done for OpenAI JSON Mode')
        
        output_format_prompt = "\nOutput in the following json string format: " + str(output_format) + "\nBe concise."
            
        my_system_prompt = str(system_prompt)
        my_user_prompt: list[str | dict] = []
        if isinstance(user_prompt, list):
            my_user_prompt.extend(user_prompt)
            my_user_prompt.append(output_format_prompt)
        else:
            my_user_prompt.append(user_prompt)
            my_user_prompt.append(output_format_prompt)
            
        res = await chat_async(my_system_prompt, my_user_prompt, response_format = {"type": "json_object"}, **kwargs)
        
        if return_as_json:
            return res
        else:
            try:
                loaded_json = json.loads(res)
            except Exception as e:
                loaded_json = {}
            return loaded_json
        
    # Otherwise, implement JSON parsing using Strict JSON
    else:
        # start off with no error message
        error_msg = ''

        # wrap the values with angle brackets and wrap keys with delimiter to encourage LLM to modify it
        new_output_format = wrap_with_angle_brackets(output_format, delimiter, 1)
        
        output_format_prompt = f'''\nOutput in the following json template: ```{new_output_format}```
Update values enclosed in <> and remove the <>. 
Your response must only be the updated json template beginning with {{ and ending with }}
Ensure the following output keys are present in the json: {' '.join(list(new_output_format.keys()))}'''

        for i in range(num_tries):
            my_system_prompt = str(system_prompt)
            my_user_prompt = []
            if isinstance(user_prompt, list):
                my_user_prompt.extend(user_prompt)
                my_user_prompt.append(output_format_prompt)
                my_user_prompt.append(error_msg)
            else:
                my_user_prompt.append(user_prompt)
                my_user_prompt.append(output_format_prompt)
                my_user_prompt.append(error_msg)
            # Use OpenAI to get a response
            res = await chat_async(my_system_prompt, my_user_prompt, **kwargs)
            
            # extract only the chunk including the opening and closing braces
            # generate the { or } if LLM has forgotten to do so
            startindex = res.find('{')
            if startindex == -1:
                startindex = 0
                res = '{' + res
            endindex = res.rfind('}')
            if endindex == -1:
                res = res + '}'
                endindex = len(res) - 1
                
            res = res[startindex: endindex+1]

            # try-catch block to ensure output format is adhered to
            try:
                # check that res is a json string
                if res[0] != '{' or res[-1] != '}':
                    raise Exception('Ensure output must be a json string beginning with { and ending with }')
                
                # do checks for keys and output format, remove escape characters so code can be run
                end_dict = await check_key_async(res, output_format, new_output_format, delimiter, delimiter_num = 1, **kwargs)
                
                assert isinstance(end_dict, dict)
                # run user defined custom checks now
                for key in end_dict:
                    if key in custom_checks:
                        for check in custom_checks[key]:
                            requirement, requirement_met, action_needed = check(end_dict[key], check_data)
                            print(f'Running check for "{requirement}" on output field of "{key}"')
                            if not requirement_met:
                                print(f'Requirement not met. Action needed: "{action_needed}"\n\n')
                                raise Exception(f'Output field of "{key}" does not meet requirement "{requirement}". Action needed: "{action_needed}"')
                            else:
                                print('Requirement met\n\n')
                if return_as_json:
                    return json.dumps(end_dict, ensure_ascii=False)
                else:
                    return end_dict

            except Exception as e:
                error_msg = f"\n\nPrevious json: {res}\njson error: {str(e)}\nFix the error."                
                print("An exception occurred:", str(e))
                print("Current invalid json format:", res)

        return {}

### Legacy Support ###
# alternative names for strict_json
strict_text_async = strict_json_async
strict_output_async = strict_json_async