from typing import Any, cast
from taskgen.agent import Agent
from taskgen.base import strict_json

from termcolor import colored

### Wrapper for Conversation that superclasses Agent
class ConversationWrapper(Agent):
    ''' This class takes an Agent and allows for conversational-based interactions with User / another Agent / Environment. Also updates persistent memory with latest information in conversation
    
    - Inputs:
        - **agent (compulsory)**: Agent. The agent we want to interact with
        - **persistent_memory**: dict. What kinds of memory the agent should have that persist over the entire conversation and their descriptions. Uses the same format as `output_format` of `strict_json`.
        - **person**: str. The name of the person you are talking to
        - **conversation**: List. The current existing conversation. Default: None
        - **num_past_conversation**: int. The number of past conversations to use for the agent
        - **verbose**: bool. Default: True. Whether to print the Agent's inner states
        
    - ConversationWrapper will automatically implement 3 new variables in `agent.shared_variables`:
        - **Persistent Memory**: The memory that will be updated as the conversation goes along, defined in persistent_dict
        - **Conversation**: The entire history of the conversation
        - **Summary of Conversation**: A summary of the current conversation
        
    - ConversationWrapper uses `chat()` which chats with the Agent and the Agent will perform actions and reply the chat message'''
    
    def __init__(self, agent: Agent, persistent_memory: dict|None = None, person = 'User', conversation = None, num_past_conversation: int = 5, verbose: bool = True):
        # Initialize the parent Agent
        super().__init__(**agent.__dict__)  # Inherit all of the attributes of the passed agent
        # Initiatlize the functions
        self.assign_functions(list(agent.function_map.values()))
        
        ## Define Additional Variables as Needed
        self.persistent_memory = persistent_memory
        self.num_past_conversation = num_past_conversation
        self.person = person
        self.verbose = verbose
        
        ''' Define some external variables for the Agent '''
        # add in the various types of memory
        self.shared_variables['Persistent Memory'] = {}
        # add in the conversation
        if conversation is None:
            self.shared_variables['Conversation'] = []
        else:
            self.shared_variables['Conversation'] = conversation
        # add in the summary of conversation
        self.shared_variables['Summary of Conversation'] = ''
    
    ## Reply the person
    def chat(self, cur_msg):
        ''' This does one chat with the person, firstly performing actions then replying the person, while updating the important memory '''
        actions_done = []
        
        ## Do actions before replying person only if there are actions other than use_llm and end_task
        my_actions = list(self.function_map.keys()) 
        if 'use_llm' in my_actions: my_actions.remove('use_llm')
        if 'end_task' in my_actions: my_actions.remove('end_task')
        if len(my_actions) > 0:
            self.reset()
            self.run(f'''Summary of Past Conversation: ```{self.shared_variables['Summary of Conversation']}```
Past Conversation: ```{self.shared_variables['Conversation'][-self.num_past_conversation:]}```
Latest input from {self.person}: ```{cur_msg}```
Use Equipped Functions other than use_llm to help answer the latest input from {self.person}''',
            )
            
            if len(self.subtasks_completed) > 0:
                actions_done = self.reply_user('Summarise Subtasks Completed in one line', verbose = False)
                print(colored(f'Actions Done: {actions_done}', 'red', attrs = ['bold']))
                print()
                self.reset()

        ## Replies the person
        res = cast(dict[str, Any], self.query(f'''Summary of Past Conversation: ```{self.shared_variables['Summary of Conversation']}```
Past Conversation: ```{self.shared_variables['Conversation'][-self.num_past_conversation:]}```
Latest Input from {self.person}: ```{cur_msg}```
Actions Done for Latest Input: ```{actions_done}```
Persistent Memory: ```{self.shared_variables['Persistent Memory']}```
Use Global Context and Conversation and Actions Done for Latest Input and and Persistent Memory as context when replying.

First think through how to reply the latest message by {self.person}, before drafting the reply.
{self.person} is not aware of Actions Done for Latest Input - include relevant information in your reply to {self.person}. Do not hallucinate actions.
Thereafter, update the Summary of Conversation''', 
                          
output_format = {"Thoughts": f"How to reply",
                 f"Reply to {self.person}": f"Your reply as {self.agent_name}",
                 "Summary of Conversation": "Summarise key points of entire conversation in at most two sentences, building on previous Summary"}))
        
        # Update the Summary of Conversation and Append the conversation
        self.shared_variables['Summary of Conversation'] = res['Summary of Conversation']
        # Append information about user and actions to conversation
        self.shared_variables['Conversation'].append(f'{self.person}: {cur_msg}')
        self.shared_variables['Conversation'].append(f'{self.agent_name}: {res[f"Reply to {self.person}"]}')
        
        ## Update Persistent Memory
        if self.persistent_memory is not None and self.persistent_memory != {}:
            persistent_memory = strict_json(f'Update all fields of Persistent Memory based on information in Additional Conversation. Current value: ```{self.shared_variables["Persistent Memory"]}```',
               f'Additional Conversation\n{self.person}: {cur_msg}\n{self.agent_name}: {res[f"Reply to {self.person}"]}',
               output_format = self.persistent_memory,
               model = self.kwargs.get('model', 'gpt-4o-mini'),
               llm = self.llm,
               verbose = self.debug)
                                                           
            self.shared_variables["Persistent Memory"] = persistent_memory
        
        if self.verbose:
            print(colored(f'Thoughts: {res["Thoughts"]}', 'green', attrs = ['bold']))
            print(colored(f'Persistent Memory: {self.shared_variables["Persistent Memory"]}', 'blue', attrs = ['bold']))
            print(colored(f'Summary of Conversation: {res["Summary of Conversation"]}', 'magenta', attrs = ['bold']))
        
        return res[f'Reply to {self.person}']

### Alternative form that does not superclass Agent. Kept here for legacy purposes ###
class ConversableAgent:
    ''' This class takes an Agent and allows for conversational-based interactions with User / another Agent / Environment. Also updates persistent memory with latest information in conversation
    
    - Inputs:
        - **agent (compulsory)**: Agent. The agent we want to interact with
        - **persistent_memory**: dict. What kinds of memory the agent should have that persist over the entire conversation and their descriptions. Uses the same format as `output_format` of `strict_json`.
        - **person**: str. The name of the person you are talking to
        - **conversation**: List. The current existing conversation. Default: None
        - **num_past_conversation**: int. The number of past conversations to use for the agent
        - **verbose**: bool. Default: True. Whether to print the Agent's inner states
        
- ConversableAgent will automatically implement 3 new variables in `agent.shared_variables`:
    - **Persistent Memory**: The memory that will be updated as the conversation goes along, defined in persistent_dict
    - **Conversation**: The entire history of the conversationn
    - **Summary of Conversation**: A summary of the current conversation
    
- ConversableAgent uses `chat()` which chats with the Agent and the Agent will perform actions and reply the chat message'''
    def __init__(self, agent: Agent, persistent_memory: dict|None = None, person = 'User', conversation = None, num_past_conversation: int = 5, verbose: bool = True):
        self.agent = agent
        self.persistent_memory = persistent_memory
        self.num_past_conversation = num_past_conversation
        self.person = person
        self.verbose = verbose
        
        ''' Define some external variables for the Agent '''
        # add in the various types of memory
        self.agent.shared_variables['Persistent Memory'] = {}
        # add in the conversation
        if conversation is None:
            self.agent.shared_variables['Conversation'] = []
        else:
            self.agent.shared_variables['Conversation'] = conversation
        # add in the summary of conversation
        self.agent.shared_variables['Summary of Conversation'] = ''
    
    ## Reply the person
    def chat(self, cur_msg):
        ''' This does one chat with the person, firstly performing actions then replying the person, while updating the important memory '''
        actions_done = []
        
        ## Do actions before replying person only if there are actions other than use_llm and end_task
        my_actions = list(self.agent.function_map.keys()) 
        if 'use_llm' in my_actions: my_actions.remove('use_llm')
        if 'end_task' in my_actions: my_actions.remove('end_task')
        if len(my_actions) > 0:
            self.agent.reset()
            self.agent.run(f'''Summary of Past Conversation: ```{self.agent.shared_variables['Summary of Conversation']}```
Past Conversation: ```{self.agent.shared_variables['Conversation'][-self.num_past_conversation:]}```
Latest input from {self.person}: ```{cur_msg}```
Use Equipped Functions other than use_llm to help answer the latest input from {self.person}''',
            )
            
            if len(self.agent.subtasks_completed) > 0:
                actions_done = self.agent.reply_user('Summarise Subtasks Completed in one line', verbose = False)
                print(colored(f'Actions Done: {actions_done}', 'red', attrs = ['bold']))
                print()
                self.agent.reset()

        ## Replies the person
        res = self.agent.query(f'''Summary of Past Conversation: ```{self.agent.shared_variables['Summary of Conversation']}```
Past Conversation: ```{self.agent.shared_variables['Conversation'][-self.num_past_conversation:]}```
Latest Input from {self.person}: ```{cur_msg}```
Actions Done for Latest Input: ```{actions_done}```
Persistent Memory: ```{self.agent.shared_variables['Persistent Memory']}```
Use Global Context and Conversation and Actions Done for Latest Input and and Persistent Memory as context when replying.

First think through how to reply the latest message by {self.person}, before drafting the reply.
{self.person} is not aware of Actions Done for Latest Input - include relevant information in your reply to {self.person}. Do not hallucinate actions.
Thereafter, update the Summary of Conversation''', 
                          
output_format = {"Thoughts": f"How to reply",
                 f"Reply to {self.person}": f"Your reply as {self.agent.agent_name}",
                 "Summary of Conversation": "Summarise key points of entire conversation in at most two sentences, building on previous Summary"})
        
        # Update the Summary of Conversation and Append the conversation
        self.agent.shared_variables['Summary of Conversation'] = res['Summary of Conversation']
        # Append information about user and actions to conversation
        self.agent.shared_variables['Conversation'].append(f'{self.person}: {cur_msg}')
        self.agent.shared_variables['Conversation'].append(f'{self.agent.agent_name}: {res[f"Reply to {self.person}"]}')
        
        ## Update Persistent Memory
        if self.persistent_memory is not None and self.persistent_memory != {}:
            persistent_memory = strict_json(f'Update all fields of Persistent Memory based on information in Additional Conversation. Current value: ```{self.agent.shared_variables["Persistent Memory"]}```',
               f'Additional Conversation\n{self.person}: {cur_msg}\n{self.agent.agent_name}: {res[f"Reply to {self.person}"]}',
               output_format = self.persistent_memory,
               model = self.agent.kwargs.get('model', 'gpt-4o-mini'),
               llm = self.agent.llm,
               verbose = self.agent.debug)
                                                           
            self.agent.shared_variables["Persistent Memory"] = persistent_memory
        
        if self.verbose:
            print(colored(f'Thoughts: {res["Thoughts"]}', 'green', attrs = ['bold']))
            print(colored(f'Persistent Memory: {self.agent.shared_variables["Persistent Memory"]}', 'blue', attrs = ['bold']))
            print(colored(f'Summary of Conversation: {res["Summary of Conversation"]}', 'magenta', attrs = ['bold']))
        
        return res[f'Reply to {self.person}']