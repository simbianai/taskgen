from taskgen import Agent, Function, Memory, Ranker
import math


# Author: @hardikmaheshwari

class Economist(Agent):
    def __init__(self):
        var_dish_price = Function(
            fn_name="dish_price",
            fn_description=''' Takes in <list_of_dish_names: list> and outputs price of each dish ''',
            output_format={'output_1': 'dict'},
            examples=None,
            external_fn=dish_price,
            is_compulsory=False)
        


        super().__init__(
            agent_name="Economist",
            agent_description='''Takes in dish names and comes up with pricing for each of them''',
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
            [var_dish_price]
        )

        self.assign_agents(
            []
        )
                        
# Supporting Functions
def dish_price(list_of_dish_names: list) -> dict:
    ''' Takes in list_of_dish_names and outputs price of each dish '''
    if not isinstance(list_of_dish_names, list):
        list_of_food_items = [list_of_dish_names]
    output = {}
    for each in list_of_dish_names:
        output[each] = '$'+str(random_number_from_string(each))
    return output


