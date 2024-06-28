from taskgen import Agent, Function, Memory, Ranker
import math


# Author: @tanchongmin
class PsychologyCounsellor(Agent):
    def __init__(self):


        super().__init__(
            agent_name="Psychology counsellor",
            agent_description='''Helps to understand and respond to User's emotion and situation. Reply user in his preferred interaction style''',
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

