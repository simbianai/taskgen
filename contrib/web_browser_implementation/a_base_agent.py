from taskgen import Agent
from utils.env_loader import load_api_key
from a_web_agent import agent as web_agent
from a_actions_agent import agent as actions_agent

load_api_key()

# Create the base agent
base_agent = Agent(
    "Base Agent",
    "Orchestrates tasks using the web and actions agents",
    verbose=True
)

# Assign the web and actions agents to the base agent
base_agent.assign_agents([web_agent, actions_agent])

# Example task for the base agent
task = "Navigate to www.example.com, take a screenshot, and click the screen at coordinates 200 and 300"

# Run the base agent to complete the task
output = base_agent.run(task)
print(f"Base Agent output: {output}")