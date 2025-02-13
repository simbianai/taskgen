import copy
from functools import wraps
import importlib
import inspect
import os
import dill as pickle
import re
import subprocess
import sys
import base64

from termcolor import colored
import requests
from taskgen.base import strict_json
from taskgen.function import Function
from taskgen.base_async import strict_json_async
from taskgen.function import AsyncFunction
from taskgen.memory import AsyncMemory, Memory
from taskgen.utils import ensure_awaitable, get_source_code_for_func


class BaseAgent:
    def __init__(
        self,
        agent_name: str = "Helpful Assistant",
        agent_description: str = "A generalist agent meant to help solve problems",
        max_subtasks: int = 5,
        summarise_subtasks_count: int = 5,
        memory_bank=None,
        shared_variables=None,
        get_global_context=None,
        global_context="",
        default_to_llm=True,
        code_action=False,
        verbose: bool = True,
        debug: bool = False,
        llm=None,
        **kwargs,
    ):
        """
        Creates an LLM-based agent using description and outputs JSON based on output_format.
        Agent does not answer in normal free-flow conversation, but outputs only concise task-based answers
        Design Philosophy:
        - Give only enough context needed to solve problem
        - Modularise components for greater accuracy of each component

        Inputs:
        - agent_name: String. Name of agent, hinting at what the agent does
        - agent_description: String. Short description of what the agent does
        - max_subtasks: Int. Default: 5. The maximum number of subtasks the agent can have
        - summarise_subtasks_count: Int. Default: 3. The maximum number of subtasks in Subtasks Completed before summary happens
        - memory_bank: class Dict[Memory]. Stores multiple types of memory for use by the agent. Customise the Memory config within the Memory class.
            - Key: `Function` (Already Implemented Natively) - Does RAG over Task -> Function mapping
            - Can add in more keys that would fit your use case. Retrieves similar items to task / overall plan (if able) for additional context in `get_next_subtasks()` and `use_llm()` function
            - For RAG over Documents, it is best done in a function of the Agent to retrieve more information when needed (so that we do not overload the Agent with information)
        - shared_variables. Dict. Default: None. Stores the variables to be shared amongst inner functions and agents.
            If not empty, will pass this dictionary by reference down to the inner agents and functions
        - get_global_context. Function. Takes in self (agent variable) and returns the additional prompt (str) to be appended to `get_next_subtask` and `use_llm`. Allows for persistent agent states to be added to the prompt
        - global_context. String. Additional global context in string form. Put in variables to substitute for shared_variables using <>
        - default_to_llm. Bool. Default: True. Whether to default to use_llm function if there is no match to other functions. If False, use_llm will not be given to Agent
        - code_action. Bool. Default: False. Whether to use code as the only action space
        - verbose: Bool. Default: True. Whether to print out intermediate thought processes of the Agent
        - debug: Bool. Default: False. Whether to debug StrictJSON messages
        - llm: Function. The llm parameter that gets passed into Function/strict_json

        Inputs (optional):
        - **kwargs: Dict. Additional arguments you would like to pass on to the strict_json function

        """
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.max_subtasks = max_subtasks
        self.summarise_subtasks_count = summarise_subtasks_count
        self.verbose = verbose
        self.default_to_llm = default_to_llm
        self.code_action = code_action
        self.get_global_context = get_global_context
        self.global_context = global_context

        self.debug = debug
        self.llm = llm

        # set shared variables
        if shared_variables is None:
            self.shared_variables = {}
        else:
            self.shared_variables = shared_variables
        self.init_shared_variables = copy.deepcopy(self.shared_variables)
        # append agent to shared variables, so that functions have access to it
        self.shared_variables["agent"] = self
        self.memory_bank = memory_bank

        # initialise the thoughts - this records the ReAct framework of Observation, Thoughts, Action at each step
        self.thoughts = []

        # reset agent's state
        self.reset()

        self.kwargs = kwargs

        # start with default of only llm as the function
        self.function_map = {}
        # stores all existing function descriptions - prevent duplicate assignment of functions
        self.fn_description_list = []

    def reset(self):
        """Resets agent state, including resetting subtasks_completed"""
        self.assign_task("No task assigned")
        self.subtasks_completed = {}
        # reset all thoughts
        self.thoughts = []

    def assign_task(self, task: str, overall_task: str = ""):
        """Assigns a new task to this agent. Also treats this as the meta agent now that we have a task assigned"""
        self.task = task
        self.overall_task = task
        # if there is a meta agent's task, add this to overall task
        if overall_task != "":
            self.overall_task = overall_task

        self.task_completed = False
        self.overall_plan = None

    async def save_agent(self, filename: str):
        """Saves the serialized agent state to a file for reuse next time"""
        if not isinstance(filename, str) or not filename.endswith('.pkl'):
            raise Exception("Filename must be a string ending with .pkl")

        # Serialize the agent state
        encoded_state = await self.serialize_agent()

        # Save the encoded state to file
        with open(filename, "w") as file:
            file.write(encoded_state)

        if self.verbose:
            print(f"Agent saved to {filename}")

    async def load_agent(self, filename: str):
        """Loads and reconstructs the agent from a saved file"""
        if not isinstance(filename, str) or not filename.endswith('.pkl'):
            raise Exception("Filename must be a string ending with .pkl")

        # Read the encoded state from file
        with open(filename, "r") as file:
            encoded_state = file.read()

        # Deserialize and reconstruct the agent
        loaded_agent = await self.__class__.deserialize_agent(encoded_state)
        
        if self.verbose:
            print(f"Agent loaded from {filename}")
        
        return loaded_agent

    def status(self):
        """Prints prettily the update of the agent's status.
        If you would want to reference any agent-specific variable, just do so directly without calling this function
        """
        print("Agent Name:", self.agent_name)
        print("Agent Description:", self.agent_description)
        print("Available Functions:", list(self.function_map.keys()))
        if len(self.shared_variables) > 0:
            print("Shared Variables:", list(self.shared_variables.keys()))
        print(colored(f"Task: {self.task}", "green", attrs=["bold"]))
        if len(self.subtasks_completed) == 0:
            print(colored("Subtasks Completed: None", "blue", attrs=["bold"]))
        else:
            print(colored("Subtasks Completed:", "black", attrs=["bold"]))
            for key, value in self.subtasks_completed.items():
                print(colored(f"Subtask: {key}", "blue", attrs=["bold"]))
                print(f"{value}\n")
        print("Is Task Completed:", self.task_completed)

    def remove_function(self, function_name: str):
        """Removes a function from the agent"""
        if function_name in self.function_map:
            function = self.function_map[function_name]
            # remove actual function from memory bank
            if function_name not in ["use_llm", "end_task"]:
                self.memory_bank["Function"].remove(function)
            # remove function description from fn_description_list
            self.fn_description_list.remove(function.fn_description)
            # remove function from function map
            del self.function_map[function_name]

    def list_functions(self, fn_list=None) -> list:
        """Returns the list of functions available to the agent. If fn_list is given, restrict the functions to only those in the list"""
        if fn_list is not None and len(fn_list) < len(self.function_map):
            if self.verbose:
                print(
                    "Filtered Function Names:",
                    ", ".join(
                        [
                            name
                            for name, function in self.function_map.items()
                            if function in fn_list
                        ]
                    ),
                )
            return [
                f"Name: {name}\n" + str(function)
                for name, function in self.function_map.items()
                if function in fn_list
            ]
        else:
            return [
                f"Name: {name}\n" + str(function)
                for name, function in self.function_map.items()
            ]

    def print_functions(self):
        """Prints out the list of functions available to the agent"""
        functions = self.list_functions()
        print("\n".join(functions))

    def add_subtask_result(self, subtask, result):
        """Adds the subtask and result to subtasks_completed
        Keep adding (num) to subtask str if there is duplicate"""
        subtask_str = str(subtask)
        count = 2

        # keep adding count until we have a unique id
        while subtask_str in self.subtasks_completed:
            subtask_str = str(subtask) + f"({count})"
            count += 1

        self.subtasks_completed[subtask_str] = result

    def remove_last_subtask(self):
        """Removes last subtask in subtask completed. Useful if you want to retrace a step"""
        if len(self.subtasks_completed) > 0:
            removed_item = self.subtasks_completed.popitem()
        if self.verbose:
            print(f"Removed last subtask from subtasks_completed: {removed_item}")

    # Alternate names
    list_function = list_functions
    list_tools = list_functions
    list_tool = list_functions
    print_function = print_functions
    print_tools = print_functions
    print_tool = print_functions
    remove_tool = remove_function

    async def serialize_agent(self) -> str:
        """Serialize essential agent state as a simple dict"""
        # Create a copy of shared_variables without the agent reference
        filtered_shared_vars = {k: v for k, v in self.shared_variables.items() if k != "agent"}

        serializable_state = {
            "agent_name": self.agent_name,
            "agent_description": self.agent_description,
            "shared_variables": filtered_shared_vars,
            "subtasks_completed": self.subtasks_completed,
            "global_context": self.global_context,
            "task": self.task,
            "task_completed": self.task_completed,
            "memory": {
                name: {"memory": bank.memory, "top_k": getattr(bank, "top_k", None)}
                for name, bank in self.memory_bank.items()
                if name != "Function"  # Skip Function memory as it contains unpicklable objects
            },
            "function_descriptions": self.fn_description_list,
        }

        return base64.b64encode(pickle.dumps(serializable_state)).decode("utf-8")

    @classmethod 
    async def deserialize_agent(cls, encoded_state: str, get_base_agent=None):
        """Reconstruct agent from serialized state"""
        state = pickle.loads(base64.b64decode(encoded_state))
        
        # Create new agent with basic state
        if get_base_agent:
            agent = await get_base_agent()  # Get pre-configured agent if provided
        else:
            agent = cls()  # Create blank agent
            
        # Restore state
        agent.agent_name = state["agent_name"]
        agent.agent_description = state["agent_description"]
        agent.shared_variables = state["shared_variables"]
        agent.shared_variables["agent"] = agent  # Restore agent reference
        agent.subtasks_completed = state["subtasks_completed"]
        agent.global_context = state["global_context"]
        agent.task = state["task"]
        agent.task_completed = state["task_completed"]
        agent.fn_description_list = state["function_descriptions"]

        # Restore memory bank data (except Function memory which is handled by get_base_agent)
        for name, memory_data in state["memory"].items():
            if name in agent.memory_bank and name != "Function":
                agent.memory_bank[name].memory = memory_data["memory"]
                if memory_data["top_k"] is not None:
                    agent.memory_bank[name].top_k = memory_data["top_k"]

        return agent


###########################
## Sync Version of Agent ##
###########################
class Agent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_memory = Memory(
            top_k=5,
            mapper=lambda x: x.fn_name + ": " + x.fn_description,
            approach="retrieve_by_ranker",
            llm=self.llm,
        )
        if self.memory_bank is None:
            self.memory_bank = {"Function": self.default_memory}
            self.memory_bank["Function"].reset()

            # adds the use llm function
        if self.default_to_llm:
            self.assign_functions(
                [
                    Function(
                        fn_name="use_llm",
                        fn_description=f"For general tasks. Used only when no other function can do the task",
                        is_compulsory=True,
                        output_format={"Output": "Output of LLM"},
                    )
                ]
            )
        # adds the end task function
        self.assign_functions(
            [
                Function(
                    fn_name="end_task",
                    fn_description="Passes the final output to the user",
                    is_compulsory=True,
                    output_format={},
                )
            ]
        )

    def query(
        self,
        query: str,
        output_format: dict,
        provide_function_list: bool = False,
        task: str = "",
    ):
        """Queries the agent with a query and outputs in output_format.
        If task is provided, we will filter the functions according to the task
        If you want to provide the agent with the context of functions available to it, set provide_function_list to True (default: False)
        If task is given, then we will use it to do RAG over functions"""

        # if we have a task to focus on, we can filter the functions (other than use_llm and end_task) by that task
        filtered_fn_list = None
        if task != "":
            # filter the functions
            filtered_fn_list = self.memory_bank["Function"].retrieve(task)

            # add back compulsory functions (default: use_llm, end_task) if present in function_map
            for name, function in self.function_map.items():
                if function.is_compulsory:
                    filtered_fn_list.append(function)

        # add in global context string and replace it with shared_variables as necessary
        global_context_string = self.global_context
        matches = re.findall(r"<(.*?)>", global_context_string)
        for match in matches:
            if match in self.shared_variables:
                global_context_string = global_context_string.replace(
                    f"<{match}>", str(self.shared_variables[match])
                )

        # add in the global context function's output
        global_context_output = (
            self.get_global_context(self) if self.get_global_context is not None else ""
        )

        global_context = ""
        # Add in global context if present
        if global_context_string != "" or global_context_output != "":
            global_context = (
                "Global Context:\n```\n"
                + global_context_string
                + "\n"
                + global_context_output
                + "```\n"
            )

        user_prompt = f"""You are an agent named {self.agent_name} with the following description: ```{self.agent_description}```\n"""
        if provide_function_list:
            user_prompt += f"You have the following Equipped Functions available:\n```{self.list_functions(filtered_fn_list)}```\n"
        user_prompt += global_context
        user_prompt += query

        res = strict_json(
            system_prompt="",
            user_prompt=user_prompt,
            output_format=output_format,
            verbose=self.debug,
            llm=self.llm,
            **self.kwargs,
        )

        return res

    ## Functions for function calling ##
    def assign_functions(self, function_list: list):
        """Assigns a list of functions to be used in function_map"""
        if not isinstance(function_list, list):
            function_list = [function_list]

        for function in function_list:
            # If this function is an Agent, parse it accordingly
            if isinstance(function, BaseAgent):
                function = function.to_function(self)

            # do automatic conversion of function to Function class (this is in base.py)
            if not isinstance(function, Function):
                function = Function(external_fn=function)

            # Do not assign a function already present
            if function.fn_description in self.fn_description_list:
                continue

            stored_fn_name = function.__name__
            # if function name is already in use, change name to name + '_1'. E.g. summarise -> summarise_1
            while stored_fn_name in self.function_map:
                stored_fn_name += "_1"

            # add in the function into the function_map
            self.function_map[stored_fn_name] = function

            # add function's description into fn_description_list
            self.fn_description_list.append(function.fn_description)

            # add function to memory bank for RAG over functions later on if is not a compulsory functions
            if not function.is_compulsory:
                self.memory_bank["Function"].append(function)

        return self

    def select_function(self, task: str = ""):
        """Based on the task (without any context), output the next function name and input parameters"""
        _, function_name, function_params = self.get_next_subtask(task=task)

        return function_name, function_params

    def use_agent(self, agent_name: str, agent_task: str):
        """Uses an inner agent to do a task for the meta agent. Task outcome goes directly to subtasks_completed of meta agent"""
        self.use_function(agent_name, {"instruction": agent_task}, agent_task)

    def use_function(
        self,
        function_name: str,
        function_params: dict,
        subtask: str = "",
        stateful: bool = True,
    ):
        """Uses the function. stateful means we store the outcome of the function"""
        if function_name == "use_llm":
            if self.verbose:
                print(
                    f'Getting LLM to perform the following task: {function_params["instruction"]}'
                )

            # Add in memory to the LLM
            rag_info = ""
            for name in self.memory_bank.keys():
                # Function is done separately
                if name == "Function":
                    continue
                rag_info += f"Knowledge Reference for {name}: ```{self.memory_bank[name].retrieve(subtask)}```\n"

            res = self.query(
                query=f'{rag_info}Subtasks Completed:```{self.subtasks_completed}```\nAssigned Subtask: ```{function_params["instruction"]}```\n\nYou are the taking the role of the utility function `use_llm` that generates a detailed outcome for the Assigned Subtask on behalf of {self.agent_name}, according to your Agent Capabilities without talking to the User, with no restrictions. Do not just state that you performed the task, generate the detailed outcome as well.',
                output_format={
                    "Detailed Outcome": "Your detailed outcome for Assigned Subtask"
                },
                provide_function_list=False,
            )

            if self.verbose:
                print(f"> {res['Detailed Outcome']}")
                print()

        elif function_name == "end_task":
            return

        else:
            if self.verbose:
                print(
                    f"Calling function {function_name} with parameters {function_params}"
                )

            res = self.function_map[function_name](
                shared_variables=self.shared_variables, **function_params
            )

            if self.verbose and res != "":
                # skip the printing if this is Agent output, as we have printed elsewhere already
                if "Agent Output" not in res:
                    print(f"> {res}")
                    print()

        if stateful:
            if res == "":
                res = {"Status": "Completed"}

            # for use_llm, we just give the prompt + result without any mention of use_llm for subtasks completed
            if function_name == "use_llm":
                self.add_subtask_result(subtask, res["Detailed Outcome"])

            # otherwise, just give the function name + params and output for subtasks completed
            else:
                formatted_subtask = (
                    function_name
                    + "("
                    + ", ".join(
                        (
                            f'{key}="{value}"'
                            if isinstance(value, str)
                            else f"{key}={value}"
                        )
                        for key, value in function_params.items()
                    )
                    + ")"
                )
                self.add_subtask_result(formatted_subtask, res)

        return res

    def get_next_subtask(self, task=""):
        """Based on what the task is and the subtasks completed, we get the next subtask, function and input parameters. Supports user-given task as well if user wants to use this function directly"""

        if task == "":
            background_info = f"Assigned Task:```\n{self.task}\n```\nSubtasks Completed: ```{self.subtasks_completed}```"

        else:
            background_info = f"Assigned Task:```\n{task}\n```\n"

        # use default agent plan if task is not given
        task = self.task if task == "" else task

        # Add in memory to the Agent
        rag_info = ""
        for name in self.memory_bank.keys():
            # Function RAG is done separately in self.query()
            if name == "Function":
                continue
            rag_info += f"Knowledge Reference for {name}: ```{self.memory_bank[name].retrieve(task)}```\n"

        # First select the Equipped Function
        res = self.query(
            query=f"""{background_info}{rag_info}\nBased on everything before, provide suitable Observation and Thoughts, and also generate the Current Subtask and the corresponding Equipped Function Name to complete a part of Assigned Task.
You are only given the Assigned Task from User with no further inputs. Only focus on the Assigned Task and do not do more than required. 
End Task if Assigned Task is completed.""",
            output_format={
                "Observation": "Reflect on what has been done in Subtasks Completed for Assigned Task",
                "Thoughts": "Brainstorm how to complete remainder of Assigned Task only given Observation",
                "Current Subtask": "What to do now in detail with all context provided that can be done by one Equipped Function for Assigned Task",
                "Equipped Function Name": "Name of Equipped Function to use for Current Subtask",
            },
            provide_function_list=True,
            task=task,
        )

        if self.verbose:
            print(
                colored(f"Observation: {res['Observation']}", "black", attrs=["bold"])
            )
            print(colored(f"Thoughts: {res['Thoughts']}", "green", attrs=["bold"]))

        # end task if equipped function is incorrect
        if res["Equipped Function Name"] not in self.function_map:
            res["Equipped Function Name"] = "end_task"

        # If equipped function is use_llm, or end_task, we don't need to do another query
        cur_function = self.function_map[res["Equipped Function Name"]]

        # Do an additional check to see if we are using code action space
        if (
            self.code_action
            and res["Equipped Function Name"] != "end_task"
            and "python_generate_and_run_code_tool" in self.function_map
        ):
            res["Equipped Function Name"] = "python_generate_and_run_code_tool"
            res["Equipped Function Inputs"] = {"instruction": res["Current Subtask"]}
        elif res["Equipped Function Name"] == "use_llm":
            res["Equipped Function Inputs"] = {"instruction": res["Current Subtask"]}
        elif res["Equipped Function Name"] == "end_task":
            res["Equipped Function Inputs"] = {}
        # Otherwise, if it is only the instruction, no type check needed, so just take the instruction
        elif (
            len(cur_function.variable_names) == 1
            and cur_function.variable_names[0].lower() == "instruction"
        ):
            res["Equipped Function Inputs"] = {"instruction": res["Current Subtask"]}

        # Otherwise, do another query to get type-checked input parameters and ensure all input fields are present
        else:
            input_format = {}
            fn_description = cur_function.fn_description
            matches = re.findall(r"<(.*?)>", fn_description)

            # do up an output format dictionary to use to get LLM to output exactly based on keys and types needed
            for match in matches:
                if ":" in match:
                    first_part, second_part = match.split(":", 1)
                    input_format[first_part] = f"A suitable value, type: {second_part}"
                else:
                    input_format[match] = "A suitable value"

            # if there is no input, then do not need LLM to extract out function's input
            if input_format == {}:
                res["Equipped Function Inputs"] = {}

            else:
                res2 = self.query(
                    query=f"""{background_info}{rag_info}\n\nCurrent Subtask: ```{res["Current Subtask"]}```\nEquipped Function Details: ```{str(cur_function)}```\nOutput suitable values for Inputs to Equipped Function to fulfil Current Subtask\nInput fields are: {list(input_format.keys())}""",
                    output_format=input_format,
                    provide_function_list=False,
                )

                # store the rest of the function parameters
                res["Equipped Function Inputs"] = res2

        # Add in output to the thoughts
        self.thoughts.append(res)

        return (
            res["Current Subtask"],
            res["Equipped Function Name"],
            res["Equipped Function Inputs"],
        )

    def summarise_subtasks_completed(self, task: str = ""):
        """Summarise the subtasks_completed list according to task"""

        output = self.reply_user(task)
        # Create a new summarised subtasks completed list
        self.subtasks_completed = {f"Current Results for '{task}'": output}

    def reply_user(self, query: str = "", stateful: bool = True, verbose: bool = True):
        """Generate a reply to the user based on the query / agent task and subtasks completed
        If stateful, also store this interaction into the subtasks_completed
        If verbose is given, can also override the verbosity of this function"""

        my_query = self.task if query == "" else query

        res = self.query(
            query=f"Subtasks Completed: ```{self.subtasks_completed}```\nAssigned Task: ```{my_query}```\nRespond to the Assigned Task using information from Global Context and Subtasks Completed only. Be factual and do not generate any new information. Be detailed and give all information available relevant for the Assigned Task in your Assigned Task Response",
            output_format={"Assigned Task Response": "Detailed Response"},
            provide_function_list=False,
        )

        res = res["Assigned Task Response"]

        if self.verbose and verbose:
            print(res)

        if stateful:
            self.add_subtask_result(my_query, res)

        return res

    def run(
        self, task: str = "", overall_task: str = "", num_subtasks: int = 0
    ) -> list:
        """Attempts to do the task using LLM and available functions
        Loops through and performs either a function call or LLM call up to num_subtasks number of times
        If overall_task is filled, then we store it to pass to the inner agents for more context
        """

        # Assign the task
        if task != "":
            ### TODO: Add in Planner here to split the task into steps if sequential generation is required
            ### Planner can also infuse diverse views if the task is more creative
            ### Planner can also be rule-based like MCTS if it is an explore-exploit RL-style problem
            ### Planner can also be rule-based shortest path algo if it is a navigation problem

            self.task_completed = False
            # If meta agent's task is here as well, assign it too
            if overall_task != "":
                self.assign_task(task, overall_task)
            else:
                self.assign_task(task)

        # check if we need to override num_steps
        if num_subtasks == 0:
            num_subtasks = self.max_subtasks

        # if task completed, then exit
        if self.task_completed:
            if self.verbose:
                print("Task already completed!\n")
                print("Subtasks completed:")
                for key, value in self.subtasks_completed.items():
                    print(f"Subtask: {key}\n{value}\n")

        else:
            # otherwise do the task
            for i in range(num_subtasks):
                # Determine next subtask, or if task is complete. Always execute if it is the first subtask
                subtask, function_name, function_params = self.get_next_subtask()
                if function_name == "end_task":
                    self.task_completed = True
                    if self.verbose:
                        print(
                            colored(
                                f"Subtask identified: End Task", "blue", attrs=["bold"]
                            )
                        )
                        print("Task completed successfully!\n")
                    break

                if self.verbose:
                    print(
                        colored(
                            f"Subtask identified: {subtask}", "blue", attrs=["bold"]
                        )
                    )

                # Execute the function for next step
                res = self.use_function(function_name, function_params, subtask)

                # Summarise Subtasks Completed if necessary
                if len(self.subtasks_completed) > self.summarise_subtasks_count:
                    print(
                        "### Auto-summarising Subtasks Completed (Change frequency via `summarise_subtasks_count` variable) ###"
                    )
                    self.summarise_subtasks_completed(
                        f"progress for {self.overall_task}"
                    )
                    print("### End of Auto-summary ###\n")

        return list(self.subtasks_completed.values())

    ## This is for Multi-Agent uses
    def to_function(self, meta_agent):
        """Converts the agent to a function so that it can be called by another agent
        The agent will take in an instruction, and output the result after processing"""

        # makes the agent appear as a function that takes in an instruction and outputs the executed instruction
        my_fn = Function(
            fn_name=self.agent_name,
            fn_description=f"Agent Description: ```{self.agent_description}```\nExecutes the given <instruction>",
            output_format={"Agent Output": "Output of instruction"},
            external_fn=Agent_External_Function(self, meta_agent),
        )

        return my_fn

    def assign_agents(self, agent_list: list):
        """Assigns a list of Agents to the main agent, passing in the meta agent as well"""
        if not isinstance(agent_list, list):
            agent_list = [agent_list]
        self.assign_functions([agent.to_function(self) for agent in agent_list])
        return self

    ###########################################################
    #### This is for agent community space (only for sync) ####
    ###########################################################
    def contribute_agent(self, author_comments=None) -> str:
        if os.environ["GITHUB_USERNAME"] is None:
            raise Exception(
                "Please set your GITHUB_USERNAME in the environment variables"
            )
        if os.environ["GITHUB_TOKEN"] is None:
            raise Exception("Please set your GITHUB_TOKEN in the environment variables")

        owner = "simbianai"
        repo = "taskgen"

        fork_url = self._create_taskgen_fork_for_user(owner, repo)
        change_tree = self._build_tree(author_comments)
        contrib_branch_name = self._commit_and_push_to_fork(fork_url, change_tree)
        pr_url = self._create_pull_request(owner, repo, fork_url, contrib_branch_name)
        return f"Pull Request created successfully at {pr_url}"

    def _create_taskgen_fork_for_user(self, owner, repo):
        url = f"https://api.github.com/repos/{owner}/{repo}/forks"
        response = requests.get(url)
        current_fork_owners = [fork["owner"]["login"] for fork in response.json()]

        if os.environ["GITHUB_USERNAME"] in current_fork_owners:
            if self.verbose:
                print(f"{os.environ['GITHUB_USERNAME']} already has a fork of taskgen")
            return [
                fork["clone_url"]
                for fork in response.json()
                if fork["owner"]["login"] == os.environ["GITHUB_USERNAME"]
            ][0]
        else:
            headers = {
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
                "X-GitHub-Api-Version": "2022-11-28",
            }

            data = {"name": "taskgen", "default_branch_only": True}

            response = requests.post(url, headers=headers, json=data)

            if response.status_code == 202:
                if self.verbose:
                    print("Fork created successfully")
                return response.json()["clone_url"]
            else:
                raise Exception(
                    f"Fork creation failed. Error: [{response.status_code}]{response.json()}"
                )

    def _build_tree_node(self, file_path, content):
        return {"path": file_path, "mode": "100644", "type": "blob", "content": content}

    def _build_tree(self, author_comments):
        agent_class_name = f"{self.agent_name.title().replace(' ', '')}_{os.environ['GITHUB_USERNAME'].replace(' ', '')}"

        directory = f"contrib/community/{agent_class_name}"
        contrib_path = f"{directory}/main.py"

        agent_code, supporting_nodes = self._get_python_rep_and_supporting_nodes(
            directory, agent_class_name, author_comments
        )
        supporting_nodes.append(self._build_tree_node(contrib_path, agent_code))

        return supporting_nodes

    def _get_python_rep_and_supporting_nodes(
        self, directory, agent_class_name, author_comments=None
    ):
        functions_code = ""
        functions_keys = []
        supporting_functions = ""

        sub_agents_imports = ""
        sub_agents_keys = []
        sub_agents_code = ""

        supporting_nodes = []

        for name, function in self.function_map.items():
            if name in "use_llm" or name in "end_task":
                continue
            if function.external_fn is not None and isinstance(
                function.external_fn, Agent_External_Function
            ):
                sub_agent_class_name = (
                    function.external_fn.agent.agent_name.title().replace(" ", "")
                )
                sub_agent_code, sub_agent_supporting_nodes = (
                    function.external_fn.agent._get_python_rep_and_supporting_nodes(
                        directory, sub_agent_class_name
                    )

                sub_agent_contrib_path = f"{directory}/{sub_agent_class_name}.py"
                supporting_nodes.extend(sub_agent_supporting_nodes)
                supporting_nodes.append(
                    self._build_tree_node(sub_agent_contrib_path, sub_agent_code)
                )

                sub_agents_imports += (
                    f"from {sub_agent_class_name} import {sub_agent_class_name}\n"
                )
                sub_agents_keys.append(f"var_agent_{sub_agent_class_name}")
                sub_agents_code += f"        var_agent_{sub_agent_class_name} = {sub_agent_class_name}()\n"
                continue

            functions_keys.append(f"var_{name}")

            function_code, external_fn_code = function.get_python_representation()
            functions_code += f"        var_{name} = {function_code}\n"
            if external_fn_code is not None:
                supporting_functions += f"{external_fn_code}\n"

        memory_bank_code = "{"
        for key, memory in self.memory_bank.items():
            memory_bank_code += f"'{key}': {memory.get_python_representation(memory != self.default_memory)},"
        memory_bank_code += "}"

        shared_variables_code = "{"
        for key, value in self.init_shared_variables.items():
            if isinstance(value, Memory):
                shared_variables_code += (
                    f"'{key}': {value.get_python_representation(True)},"
                )
            else:
                shared_variables_code += f"'{key}': {value},"
        shared_variables_code += "}"

        get_global_context_ref = None
        if self.get_global_context:
            if (
                inspect.isfunction(self.get_global_context)
                and self.get_global_context.__name__ == "<lambda>"
            ):
                get_global_context_ref = get_source_code_for_func(
                    self.get_global_context
                )
            else:
                get_global_context_ref = self.get_global_context.__name__
                supporting_functions += (
                    f"{get_source_code_for_func(self.get_global_context)}\n"
                )

        agent_code = f"""from taskgen import Agent, Function, Memory, Ranker
import math
{sub_agents_imports}

# Author: @{os.environ['GITHUB_USERNAME']}
{"# Author Comments: " + author_comments if author_comments else ''}
class {agent_class_name}(Agent):
    def __init__(self):
{functions_code}
{sub_agents_code}
        super().__init__(
            agent_name="{self.agent_name}",
            agent_description='''{self.agent_description}''',
            max_subtasks={self.max_subtasks},
            summarise_subtasks_count={self.summarise_subtasks_count},
            memory_bank={memory_bank_code},
            shared_variables={shared_variables_code},
            get_global_context={get_global_context_ref},
            global_context='''{self.global_context}''',
            default_to_llm={self.default_to_llm},
            code_action={self.code_action},
            verbose={self.verbose},
            debug={self.debug}
        )

        self.assign_functions(
            [{','.join(functions_keys)}]
        )

        self.assign_agents(
            [{','.join(sub_agents_keys)}]
        )
                        
# Supporting Functions
{supporting_functions}
"""
        return agent_code, supporting_nodes

    def _get_current_branches(self, owner, repo):
        headers = {
            "Authorization": f'token {os.environ["GITHUB_TOKEN"]}',
            "Accept": "application/vnd.github.v3+json",
        }
        url = f"https://api.github.com/repos/{owner}/{repo}/branches"
        response = requests.get(url, headers=headers)

        all_branches = []
        if response.status_code == 200:
            branches = response.json()
            all_branches.extend([branch["name"] for branch in branches])
        else:
            raise Exception(f"Failed to fetch branches: {response.status_code}")

        while "next" in response.links.keys():
            response = requests.get(response.links["next"]["url"], headers=headers)
            if response.status_code == 200:
                branches = response.json()
                all_branches.extend([branch["name"] for branch in branches])
            else:
                raise Exception(
                    f"Failed to fetch additional branches: {response.status_code}"
                )

        return all_branches

    def _create_branch(self, branch_name, owner, repo):
        base_branch = "main"
        headers = {
            "Authorization": f'token {os.environ["GITHUB_TOKEN"]}',
            "Accept": "application/vnd.github.v3+json",
        }

        # 1. Get the SHA of the latest commit on the base branch
        base_url = (
            f"https://api.github.com/repos/{owner}/{repo}/git/refs/heads/{base_branch}"
        )
        r = requests.get(base_url, headers=headers)
        if r.status_code != 200:
            raise Exception(f"Failed to get base branch info: {r.status_code}")
        base_sha = r.json()["object"]["sha"]

        # 2. Create the new branch
        create_url = f"https://api.github.com/repos/{owner}/{repo}/git/refs"
        data = {"ref": f"refs/heads/{branch_name}", "sha": base_sha}

        r = requests.post(create_url, headers=headers, json=data)

        if r.status_code == 201:
            if self.verbose:
                print(f"Successfully created new branch: {branch_name}")
        else:
            raise Exception(f"Failed to create branch: {r.status_code}")

    def _commit_and_push_to_fork(self, fork_url: str, change_tree: list):
        agent_class_name = change_tree[-1]["path"].split("/")[-2]
        contrib_branch_name = f"contribute-agent-{agent_class_name}"

        owner = fork_url.split("/")[-2]
        repo = fork_url.split("/")[-1][:-4]

        if contrib_branch_name in self._get_current_branches(owner=owner, repo=repo):
            raise Exception(
                f"Branch {contrib_branch_name} already exists. Please delete the branch and try again."
            )

        # Create branch
        self._create_branch(branch_name=contrib_branch_name, owner=owner, repo=repo)

        headers = {
            "Authorization": f'token {os.environ["GITHUB_TOKEN"]}',
            "Accept": "application/vnd.github.v3+json",
        }

        # Get the latest commit SHA
        url = f"https://api.github.com/repos/{owner}/{repo}/git/refs/heads/{contrib_branch_name}"
        r = requests.get(url, headers=headers)
        sha_latest_commit = r.json()["object"]["sha"]

        # Create a new tree
        url = f"https://api.github.com/repos/{owner}/{repo}/git/trees"
        data = {"base_tree": sha_latest_commit, "tree": change_tree}
        r = requests.post(url, headers=headers, json=data)
        sha_new_tree = r.json()["sha"]

        # Create a new commit
        url = f"https://api.github.com/repos/{owner}/{repo}/git/commits"
        data = {
            "message": f"Contribute agent: {self.agent_name}",
            "tree": sha_new_tree,
            "parents": [sha_latest_commit],
        }
        r = requests.post(url, headers=headers, json=data)
        sha_new_commit = r.json()["sha"]

        # Update the reference
        url = f"https://api.github.com/repos/{owner}/{repo}/git/refs/heads/{contrib_branch_name}"
        data = {"sha": sha_new_commit}
        r = requests.patch(url, headers=headers, json=data)

        if r.status_code == 200:
            if self.verbose:
                print("Successfully pushed changes to remote branch.")
        else:
            raise Exception("Failed to push changes.")

        return contrib_branch_name

    def _create_pull_request(self, owner, repo, fork_url, contrib_branch_name):
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls"

        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        data = {
            "title": f"Contribute agent: {self.agent_name}",
            "body": f"""Agent Details:
- Agent Name: {self.agent_name}
- Agent Description: {self.agent_description}
""",
            "head_repo": f"{fork_url.split('/')[-1].split('.')[0]}",
            "head": f"{os.environ['GITHUB_USERNAME']}:{contrib_branch_name}",
            "base": "main",
        }

        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 201:
            if self.verbose:
                print("Pull request created successfully")
            return response.json()["html_url"]
        else:
            raise Exception(f"Error: {response.status_code}, {response.json()}")

    @classmethod
    def load_community_agent(cls, agent_name: str):
        # Convert the agent name to the expected class name
        agent_class_name = agent_name.title().replace(" ", "")

        # Construct the full directory path where the agent and its dependencies reside
        directory = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../contrib/community",
            agent_class_name,
        )

        # Construct the full path to the main.py file within the agent's directory
        module_path = os.path.join(directory, "main.py")

        # Check if the module file exists, raise an exception if not
        if not os.path.exists(module_path):
            raise Exception(f"Agent {agent_name} does not exist in the community")

        # Add the directory to sys.path to ensure dependencies can be imported
        if directory not in sys.path:
            sys.path.insert(0, directory)

        # Load the module from the given file location
        spec = importlib.util.spec_from_file_location(agent_class_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Remove the directory from sys.path after loading to clean up
        try:
            sys.path.remove(directory)
        except ValueError:
            pass  # Handle the case where the directory was not added or already removed

        # Check if the module has the expected class, and instantiate it if it does
        if hasattr(module, agent_class_name):
            return getattr(module, agent_class_name)()
        else:
            raise AttributeError(
                f"The class {agent_class_name} does not exist in the module {module_path}"
            )

    ## Function aliaises
    assign_function = assign_functions
    assign_tool = assign_functions
    assign_tools = assign_functions
    select_tool = select_function
    use_tool = use_function
    assign_agent = assign_agents


############################
## Async Version of Agent ##
############################


class AsyncAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_memory = AsyncMemory(
            top_k=5,
            mapper=lambda x: (x.fn_name or "") + ": " + (x.fn_description or ""),
            approach="retrieve_by_ranker",
            llm=self.llm,
        )
        if self.memory_bank is None:
            self.memory_bank = {"Function": self.default_memory}
            self.memory_bank["Function"].reset()
        if not isinstance(self.memory_bank["Function"], AsyncMemory):
            raise Exception("Sync memory not allowed for Async Agent")
        if self.default_to_llm:
            self.assign_functions(
                [
                    AsyncFunction(
                        fn_name="use_llm",
                        fn_description=f"For general tasks. Used only when no other function can do the task",
                        is_compulsory=True,
                        output_format={"Output": "Output of LLM"},
                    )
                ]
            )
        # adds the end task function
        self.assign_functions(
            [
                AsyncFunction(
                    fn_name="end_task",
                    fn_description="Passes the final output to the user",
                    is_compulsory=True,
                    output_format={},
                )
            ]
        )

    async def query(
        self,
        query: str,
        output_format: dict,
        provide_function_list: bool = False,
        task: str = "",
    ):
        """Queries the agent with a query and outputs in output_format.
        If task is provided, we will filter the functions according to the task
        If you want to provide the agent with the context of functions available to it, set provide_function_list to True (default: False)
        If task is given, then we will use it to do RAG over functions"""

        # if we have a task to focus on, we can filter the functions (other than use_llm and end_task) by that task
        filtered_fn_list = None
        if task != "":
            # filter the functions
            filtered_fn_list = await self.memory_bank["Function"].retrieve(task)

            # add back compulsory functions (default: use_llm, end_task) if present in function_map
            for name, function in self.function_map.items():
                if function.is_compulsory:
                    filtered_fn_list.append(function)

        # add in global context string and replace it with shared_variables as necessary
        global_context_string = self.global_context
        matches = re.findall(r"<(.*?)>", global_context_string)
        for match in matches:
            if match in self.shared_variables:
                global_context_string = global_context_string.replace(
                    f"<{match}>", str(self.shared_variables[match])
                )

        # add in the global context function's output
        global_context_output = (
            self.get_global_context(self) if self.get_global_context is not None else ""
        )

        global_context = ""
        # Add in global context if present
        if global_context_string != "" or global_context_output != "":
            global_context = (
                "Global Context:\n```\n"
                + global_context_string
                + "\n"
                + global_context_output
                + "```\n"
            )

        user_prompt = f"""You are an agent named {self.agent_name} with the following description: ```{self.agent_description}```\n"""
        if provide_function_list:
            user_prompt += f"You have the following Equipped Functions available:\n```{self.list_functions(filtered_fn_list)}```\n"
        user_prompt += global_context
        user_prompt += query

        res = await strict_json_async(
            system_prompt="",
            user_prompt=user_prompt,
            output_format=output_format,
            verbose=self.debug,
            llm=self.llm,
            **self.kwargs,
        )

        return res

    ## Functions for function calling ##
    def assign_functions(self, function_list: list):
        """Assigns a list of functions to be used in function_map"""
        if not isinstance(function_list, list):
            function_list = [function_list]

        for function in function_list:
            # If this function is an Agent, parse it accordingly
            if isinstance(function, BaseAgent):
                function = function.to_function(self)

            # do automatic conversion of function to Function class (this is in base.py)
            if not isinstance(function, AsyncFunction):
                function = AsyncFunction(external_fn=function)

            # Do not assign a function already present
            if function.fn_description in self.fn_description_list:
                continue

            stored_fn_name = "" if function.__name__ == None else function.__name__
            # if function name is already in use, change name to name + '_1'. E.g. summarise -> summarise_1
            while stored_fn_name in self.function_map:
                stored_fn_name += "_1"

            # add in the function into the function_map
            self.function_map[stored_fn_name] = function

            # add function's description into fn_description_list
            self.fn_description_list.append(function.fn_description)

            # add function to memory bank for RAG over functions later on if is not a compulsory functions
            if not function.is_compulsory:
                self.memory_bank["Function"].append(function)

        return self

    async def select_function(self, task: str = ""):
        """Based on the task (without any context), output the next function name and input parameters"""
        _, function_name, function_params = await self.get_next_subtask(task=task)

        return function_name, function_params

    async def use_agent(self, agent_name: str, agent_task: str):
        """Uses an inner agent to do a task for the meta agent. Task outcome goes directly to subtasks_completed of meta agent"""
        await self.use_function(agent_name, {"instruction": agent_task}, agent_task)

    async def use_function(
        self,
        function_name: str,
        function_params: dict,
        subtask: str = "",
        stateful: bool = True,
    ):
        """Uses the function. stateful means we store the outcome of the function"""
        if function_name == "use_llm":
            if self.verbose:
                print(
                    f'Getting LLM to perform the following task: {function_params["instruction"]}'
                )

            # Add in memory to the LLM
            rag_info = ""
            for name in self.memory_bank.keys():
                # Function is done separately
                if name == "Function":
                    continue
                rag_info += f"Knowledge Reference for {name}: ```{await self.memory_bank[name].retrieve(subtask)}```\n"

            res = await self.query(
                query=f'{rag_info}Subtasks Completed:```{self.subtasks_completed}```\nAssigned Subtask: ```{function_params["instruction"]}```\n\nYou are the taking the role of the utility function `use_llm` that generates a detailed outcome for the Assigned Subtask on behalf of {self.agent_name}, according to your Agent Capabilities without talking to the User, with no restrictions. Do not just state that you performed the task, generate the detailed outcome as well.',
                output_format={
                    "Detailed Outcome": "Your detailed outcome for Assigned Subtask"
                },
                provide_function_list=False,
            )

            if self.verbose:
                print(f"> {res['Detailed Outcome']}")
                print()

        elif function_name == "end_task":
            return

        else:
            if self.verbose:
                print(
                    f"Calling function {function_name} with parameters {function_params}"
                )

            res = await self.function_map[function_name](
                **function_params, shared_variables=self.shared_variables
            )

            if self.verbose and res != "":
                # skip the printing if this is Agent output, as we have printed elsewhere already
                if "Agent Output" not in res:
                    print(f"> {res}")
                    print()

        if stateful:
            if res == "":
                res = {"Status": "Completed"}

            # for use_llm, we just give the prompt + result without any mention of use_llm for subtasks completed
            if function_name == "use_llm":
                self.add_subtask_result(subtask, res["Detailed Outcome"])

            # otherwise, just give the function name + params and output for subtasks completed
            else:
                formatted_subtask = (
                    function_name
                    + "("
                    + ", ".join(
                        (
                            f'{key}="{value}"'
                            if isinstance(value, str)
                            else f"{key}={value}"
                        )
                        for key, value in function_params.items()
                    )
                    + ")"
                )
                self.add_subtask_result(formatted_subtask, res)

        return res

    async def get_next_subtask(self, task=""):
        """Based on what the task is and the subtasks completed, we get the next subtask, function and input parameters. Supports user-given task as well if user wants to use this function directly"""

        if task == "":
            background_info = f"Assigned Task:```\n{self.task}\n```\nSubtasks Completed: ```{self.subtasks_completed}```"

        else:
            background_info = f"Assigned Task:```\n{task}\n```\n"

        # use default agent plan if task is not given
        task = self.task if task == "" else task

        # Add in memory to the Agent
        rag_info = ""
        for name in self.memory_bank.keys():
            # Function RAG is done separately in self.query()
            if name == "Function":
                continue
            rag_info += f"Knowledge Reference for {name}: ```{await self.memory_bank[name].retrieve(task)}```\n"

        # First select the Equipped Function
        res = await self.query(
            query=f"""{background_info}{rag_info}\nBased on everything before, provide suitable Observation and Thoughts, and also generate the Current Subtask and the corresponding Equipped Function Name to complete a part of Assigned Task.
You are only given the Assigned Task from User with no further inputs. Only focus on the Assigned Task and do not do more than required.
End Task if Assigned Task is completed.""",
            output_format={
                "Observation": "Reflect on what has been done in Subtasks Completed for Assigned Task",
                "Thoughts": "Brainstorm how to complete remainder of Assigned Task only given Observation",
                "Current Subtask": "What to do now in detail with all context provided that can be done by one Equipped Function for Assigned Task",
                "Equipped Function Name": "Name of Equipped Function to use for Current Subtask",
            },
            provide_function_list=True,
            task=task,
        )

        if self.verbose:
            print(
                colored(f"Observation: {res['Observation']}", "black", attrs=["bold"])
            )
            print(colored(f"Thoughts: {res['Thoughts']}", "green", attrs=["bold"]))

        # end task if equipped function is incorrect
        if res["Equipped Function Name"] not in self.function_map:
            res["Equipped Function Name"] = "end_task"

        # If equipped function is use_llm, or end_task, we don't need to do another query
        cur_function = self.function_map[res["Equipped Function Name"]]

        # Do an additional check to see if we should use code
        if (
            self.code_action
            and res["Equipped Function Name"] != "end_task"
            and "python_generate_and_run_code_tool" in self.function_map
        ):
            res["Equipped Function Name"] = "python_generate_and_run_code_tool"
            res["Equipped Function Inputs"] = {"instruction": res["Current Subtask"]}
        elif res["Equipped Function Name"] == "use_llm":
            res["Equipped Function Inputs"] = {"instruction": res["Current Subtask"]}
        elif res["Equipped Function Name"] == "end_task":
            res["Equipped Function Inputs"] = {}
        # Otherwise, if it is only the instruction, no type check needed, so just take the instruction
        elif (
            len(cur_function.variable_names) == 1
            and cur_function.variable_names[0].lower() == "instruction"
        ):
            res["Equipped Function Inputs"] = {"instruction": res["Current Subtask"]}

        # Otherwise, do another query to get type-checked input parameters and ensure all input fields are present
        else:
            input_format = {}
            fn_description = cur_function.fn_description
            matches = re.findall(r"<(.*?)>", fn_description)

            # do up an output format dictionary to use to get LLM to output exactly based on keys and types needed
            for match in matches:
                if ":" in match:
                    first_part, second_part = match.split(":", 1)
                    input_format[first_part] = f"A suitable value, type: {second_part}"
                else:
                    input_format[match] = "A suitable value"

            # if there is no input, then do not need LLM to extract out function's input
            if input_format == {}:
                res["Equipped Function Inputs"] = {}

            else:
                res2 = await self.query(
                    query=f"""{background_info}{rag_info}\n\nCurrent Subtask: ```{res["Current Subtask"]}```\nEquipped Function Details: ```{str(cur_function)}```\Output suitable values for Inputs to Equipped Function to fulfil Current Subtask\nInput fields are: {list(input_format.keys())}""",
                    output_format=input_format,
                    provide_function_list=False,
                )

                # store the rest of the function parameters
                res["Equipped Function Inputs"] = res2

        # Add in output to the thoughts
        self.thoughts.append(res)

        return (
            res["Current Subtask"],
            res["Equipped Function Name"],
            res["Equipped Function Inputs"],
        )

    async def summarise_subtasks_completed(self, task: str = ""):
        """Summarise the subtasks_completed list according to task"""

        output = await self.reply_user(task)
        # Create a new summarised subtasks completed list
        self.subtasks_completed = {f"Current Results for '{task}'": output}

    async def reply_user(
        self, query: str = "", stateful: bool = True, verbose: bool = True
    ):
        """Generate a reply to the user based on the query / agent task and subtasks completed
        If stateful, also store this interaction into the subtasks_completed
        If verbose is given, can also override the verbosity of this function"""

        my_query = self.task if query == "" else query

        res = await self.query(
            query=f"Subtasks Completed: ```{self.subtasks_completed}```\nAssigned Task: ```{my_query}```\nRespond to the Assigned Task in detail using information from Global Context and Subtasks Completed only. Be factual and do not generate any new information. Be detailed and give all information available relevant for the Assigned Task in your Assigned Task Response",
            output_format={"Assigned Task Response": "Detailed Response"},
            provide_function_list=False,
        )

        res = res["Assigned Task Response"]

        if self.verbose and verbose:
            print(res)

        if stateful:
            self.add_subtask_result(my_query, res)

        return res

    async def run(
        self, task: str = "", overall_task: str = "", num_subtasks: int = 0
    ) -> list:
        """Attempts to do the task using LLM and available functions
        Loops through and performs either a function call or LLM call up to num_subtasks number of times
        If overall_task is filled, then we store it to pass to the inner agents for more context
        """

        # Assign the task
        if task != "":
            ### TODO: Add in Planner here to split the task into steps if sequential generation is required
            ### Planner can also infuse diverse views if the task is more creative
            ### Planner can also be rule-based like MCTS if it is an explore-exploit RL-style problem
            ### Planner can also be rule-based shortest path algo if it is a navigation problem

            self.task_completed = False
            # If meta agent's task is here as well, assign it too
            if overall_task != "":
                self.assign_task(task, overall_task)
            else:
                self.assign_task(task)

        # check if we need to override num_steps
        if num_subtasks == 0:
            num_subtasks = self.max_subtasks

        # if task completed, then exit
        if self.task_completed:
            if self.verbose:
                print("Task already completed!\n")
                print("Subtasks completed:")
                for key, value in self.subtasks_completed.items():
                    print(f"Subtask: {key}\n{value}\n")

        else:
            # otherwise do the task
            for i in range(num_subtasks):
                # Determine next subtask, or if task is complete. Always execute if it is the first subtask
                subtask, function_name, function_params = await self.get_next_subtask()
                if function_name == "end_task":
                    self.task_completed = True
                    if self.verbose:
                        print(
                            colored(
                                f"Subtask identified: End Task", "blue", attrs=["bold"]
                            )
                        )
                        print("Task completed successfully!\n")
                    break

                if self.verbose:
                    print(
                        colored(
                            f"Subtask identified: {subtask}", "blue", attrs=["bold"]
                        )
                    )

                # Execute the function for next step
                res = await self.use_function(function_name, function_params, subtask)

                # Summarise Subtasks Completed if necessary
                if len(self.subtasks_completed) > self.summarise_subtasks_count:
                    print(
                        "### Auto-summarising Subtasks Completed (Change frequency via `summarise_subtasks_count` variable) ###"
                    )
                    await self.summarise_subtasks_completed(
                        f"progress for {self.overall_task}"
                    )
                    print("### End of Auto-summary ###\n")

        return list(self.subtasks_completed.values())

    ## This is for Multi-Agent uses
    def to_function(self, meta_agent):
        """Converts the agent to a function so that it can be called by another agent
        The agent will take in an instruction, and output the result after processing"""

        # makes the agent appear as a function that takes in an instruction and outputs the executed instruction
        my_fn = AsyncFunction(
            fn_name=self.agent_name,
            fn_description=f"Agent Description: ```{self.agent_description}```\nExecutes the given <instruction>",
            output_format={"Agent Output": "Output of instruction"},
            external_fn=to_function(
                Async_Agent_External_Function(self, meta_agent).__call__
            ),
        )

        return my_fn

    def assign_agents(self, agent_list: list):
        """Assigns a list of Agents to the main agent, passing in the meta agent as well"""
        if not isinstance(agent_list, list):
            agent_list = [agent_list]
        self.assign_functions([agent.to_function(self) for agent in agent_list])
        return self

    ## Function aliaises
    assign_function = assign_functions
    assign_tool = assign_functions
    assign_tools = assign_functions
    select_tool = select_function
    use_tool = use_function
    assign_agent = assign_agents


###########################################
### This is to wrap Agents as Functions ###
###########################################


class Base_Agent_External_Function:
    """Creates a Function-based version of the agent"""

    def __init__(self, agent: Agent, meta_agent: Agent):
        """Retains the instance of the agent as an internal variable"""
        self.agent = agent
        self.meta_agent = meta_agent


class Agent_External_Function(Base_Agent_External_Function):
    """Creates a Function-based version of the agent"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, instruction: str):
        """Calls the Inner Agent to perform an instruction. The outcome of the agent goes directly into subtasks_completed
        Returns the Inner Agent's reply"""
        # make a deep copy so we do not affect the original agent
        if self.agent.verbose:
            print(f"\n### Start of Inner Agent: {self.agent.agent_name} ###")
        agent_copy = copy.deepcopy(self.agent)

        # take the shared variables from the meta agent
        agent_copy.shared_variables = self.meta_agent.shared_variables
        agent_copy.shared_variables["agent"] = agent_copy

        # provide the subtasks completed and debug capabilities to the inner agents too
        agent_copy.reset()
        agent_copy.debug = self.meta_agent.debug
        if len(self.meta_agent.subtasks_completed) > 0:
            agent_copy.global_context += (
                f"Related Subtasks Completed: {self.meta_agent.subtasks_completed}"
            )
        agent_copy.subtasks_completed = {}

        output = agent_copy.run(instruction, self.meta_agent.overall_task)

        # append result of inner agent to meta agent
        agent_copy.verbose = False
        agent_reply = agent_copy.reply_user()

        if self.agent.verbose:
            print(
                colored(
                    f"###\nReply from {self.agent.agent_name} to {self.meta_agent.agent_name}:\n{agent_reply}\n###\n",
                    "magenta",
                    attrs=["bold"],
                )
            )
            print(f"### End of Inner Agent: {self.agent.agent_name} ###\n")

        # sets back the original agent in shared variables
        self.meta_agent.shared_variables["agent"] = self.meta_agent

        return agent_reply


class Async_Agent_External_Function(Base_Agent_External_Function):
    """Creates a Function-based version of the agent"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.agent, AsyncAgent):
            raise TypeError("Expected Async agent but provided with sync Agent")

    async def __call__(self, instruction: str):
        """Calls the Inner Agent to perform an instruction. The outcome of the agent goes directly into subtasks_completed
        Returns the Inner Agent's reply"""
        # make a deep copy so we do not affect the original agent
        if self.agent.verbose:
            print(f"\n### Start of Inner Agent: {self.agent.agent_name} ###")
        agent_copy = copy.deepcopy(self.agent)
        agent_copy.global_context = self.agent.global_context
        # take the shared variables from the meta agent
        agent_copy.shared_variables = self.meta_agent.shared_variables
        agent_copy.shared_variables["agent"] = agent_copy

        # provide the subtasks completed and debug capabilities to the inner agents too
        agent_copy.reset()
        agent_copy.debug = self.meta_agent.debug
        if len(self.meta_agent.subtasks_completed) > 0:
            agent_copy.global_context += (
                f"Related Subtasks Completed: {self.meta_agent.subtasks_completed}"
            )
        agent_copy.subtasks_completed = {}

        output = await agent_copy.run(instruction, self.meta_agent.overall_task)

        # append result of inner agent to meta agent
        agent_copy.verbose = False
        agent_reply = await agent_copy.reply_user()

        if self.agent.verbose:
            print(
                colored(
                    f"###\nReply from {self.agent.agent_name} to {self.meta_agent.agent_name}:\n{agent_reply}\n###\n",
                    "magenta",
                    attrs=["bold"],
                )
            )
            print(f"### End of Inner Agent: {self.agent.agent_name} ###\n")

        # sets back the original agent in shared variables
        self.meta_agent.shared_variables["agent"] = self.meta_agent

        return agent_reply


def to_function(method, **fixed_params):
    @wraps(method)
    async def wrapper(*args, **kwargs):
        # Merge the fixed params with the kwargs passed during call
        all_kwargs = {**fixed_params, **kwargs}
        # Execute the method and store result in shared_variables
        result = await method(*args, **all_kwargs)
        return result

    return wrapper
