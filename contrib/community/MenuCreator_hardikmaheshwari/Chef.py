from taskgen import Agent, Function, Memory, Ranker
import math


# Author: @hardikmaheshwari

class Chef(Agent):
    def __init__(self):


        super().__init__(
            agent_name="Chef",
            agent_description='''Takes in dish names and generates ingredients for each of them. Does not generate prices.''',
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
            []
        )

        self.assign_agents(
            []
        )
                        
# Supporting Functions

