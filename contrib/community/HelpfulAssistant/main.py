from taskgen import Agent, Function, Memory, Ranker
import math


# Author: @hardikmaheshwari
class HelpfulAssistant(Agent):
    def __init__(self):
        var_binary_to_decimal = Function(
            fn_name="binary_to_decimal",
            fn_description='''Convert input <x: a binary number in base 2> to base 10''',
            output_format={'output1': 'x in base 10'},
            examples=None,
            external_fn=binary_to_decimal,
            is_compulsory=False)
        
        var_sentence_with_objects_entities_emotion = Function(
            fn_name="sentence_with_objects_entities_emotion",
            fn_description='''Output a sentence with <obj> and <entity> in the style of <emotion>''',
            output_format={'output': 'sentence'},
            examples=None,
            external_fn=None,
            is_compulsory=False)
        


        super().__init__(
            agent_name="Helpful assistant",
            agent_description='''You are a generalist agent''',
            max_subtasks=5,
            summarise_subtasks_count=5,
            memory_bank={'Function': Memory(memory=[], top_k=5, mapper=lambda x: x.fn_name + ': ' + x.fn_description, approach='retrieve_by_ranker', ranker=Ranker(model='text-embedding-3-small', ranking_fn=None)),},
            shared_variables={},
            get_global_context=None,
            global_context='''''',
            default_to_llm=True,
            verbose=True,
            debug=False
        )

        self.assign_functions(
            [var_binary_to_decimal,var_sentence_with_objects_entities_emotion]
        )

        self.assign_agents(
            []
        )
                        
# Supporting Functions
def binary_to_decimal(x):
    return int(str(x), 2)


