from scripts.actions import Create_a_new_instance_of_the_Chrome_driver, Navigate_to_a_web_page

from utils.env_loader import load_api_key

from taskgen import Agent, Function

api_key = load_api_key()

# LLM-based function
llm_function = Function(
    fn_description="Generate a greeting message for <name>",
    output_format={"output": "greeting message"},
)

# Define functions for the agent
fn_list = [
    Function(
        fn_description="Create a new instance of the Chrome driver <x: int>",
        external_fn=Create_a_new_instance_of_the_Chrome_driver,
        output_format={"Output": "Chrome driver instance"},
    ),
    Function(
        fn_description="Navigate to a web page <url: str>",
        external_fn=Navigate_to_a_web_page,
        output_format={"Output": "Navigation status"},
    ),
]

# Create the agent
agent = Agent(
    "Screen Interaction Agent",
    "Performs screen interaction actions",
    shared_variables={'Driver': ''},
    verbose=True
).assign_functions(fn_list)

# see the auto-generated names of your functions :)
agent.list_functions()

# Create a new instance of the Chrome driver
output = agent.run("Create 1 new instance of the Chrome driver then navigate to www.example.com web page")
print(f"Agent output: {output}")  # Add this line to print the agent's output

