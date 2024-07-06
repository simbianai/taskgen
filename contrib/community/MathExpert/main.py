from taskgen import Agent, Function, Memory, Ranker
import math


# Author: @hardikmaheshwari
# Author Comments: This agent can be leveraged for addition functionality
class MathExpert(Agent):
    def __init__(self):
        var_add = Function(
            fn_name="add",
            fn_description='''Takes in <x: int> and <y: int> and returns the sum''',
            output_format={'output_1': 'int'},
            examples=None,
            external_fn=add,
            is_compulsory=False)
        


        super().__init__(
            agent_name="Math Expert",
            agent_description='''Does Math very well''',
            max_subtasks=5,
            summarise_subtasks_count=5,
            memory_bank={'Function': Memory(memory=[], top_k=5, mapper=lambda x: x.fn_name + ': ' + x.fn_description, approach='retrieve_by_ranker', ranker=Ranker(model='text-embedding-3-small', ranking_fn=None)),},
            shared_variables={},
            get_global_context=None,
            global_context='''''',
            default_to_llm=True,
            code_action=False,
            verbose=True,
            debug=False
        )

        self.assign_functions(
            [var_add]
        )

        self.assign_agents(
            []
        )
                        
# Supporting Functions
def add(x: int, y: int) -> int:
    '''Takes in x and y and returns the sum'''
    return x+y


