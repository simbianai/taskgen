from taskgen import Agent

# Author: @hardikmaheshwari
class HelpfulAssistant(Agent):
    def __init__(self):
        super().__init__(agent_name="Helpful assistant",
                        agent_description="You are a generalist agent")
