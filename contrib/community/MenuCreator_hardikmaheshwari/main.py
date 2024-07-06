from taskgen import Agent, Function, Memory, Ranker
import math
from Chef import Chef
from Boss import Boss
from CreativeWriter import CreativeWriter
from Economist import Economist


# Author: @hardikmaheshwari
# Author Comments: Leverages sub agent to generate menu for a restraunt
class MenuCreator_hardikmaheshwari(Agent):
    def __init__(self):

        var_agent_Chef = Chef()
        var_agent_Boss = Boss()
        var_agent_CreativeWriter = CreativeWriter()
        var_agent_Economist = Economist()

        super().__init__(
            agent_name="Menu Creator",
            agent_description='''Creates a menu for a restaurant. Menu item includes Name, Description, Ingredients, Pricing.''',
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
            [var_agent_Chef,var_agent_Boss,var_agent_CreativeWriter,var_agent_Economist]
        )
                        
# Supporting Functions

