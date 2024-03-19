import os
from datetime import datetime

import pyautogui

from utils.create_screenshot_folder import create_screenshots_folder
from utils.env_loader import load_api_key

from taskgen import Agent, Function

api_key = load_api_key()

# LLM-based function
llm_function = Function(
    fn_description="Generate a greeting message for <name>",
    output_format={"output": "greeting message"},
)

def take_screenshot(shared_variables, x: int):
    screenshots_folder = create_screenshots_folder()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_filename = f"screenshot_{timestamp}.png"
    screenshot_path = os.path.join(screenshots_folder, screenshot_filename)
    screenshot = pyautogui.screenshot()
    screenshot.save(screenshot_path)
    return {"Output": f"Screenshot saved at: {screenshot_path}"}


def click_screen(shared_variables, x: int, y: int):
    pyautogui.click(x, y)
    return {"Output": f"Clicked the screen at coordinates ({x}, {y})"}


# Define functions for the agent
fn_list = [
    Function(
        fn_description="Clicks the screen at coordinates <x: int> and <y: int>",
        external_fn=click_screen,
        output_format={"Output": "Action status"},
    ),
    Function(
        fn_description="Takes a screenshot and saves it to a file <x: int>",
        external_fn=take_screenshot,
        output_format={"Output": "Screenshot save path"},
    ),
]

# Create the agent
agent = Agent(
    "Screen Interaction Agent",
    "Performs screen interaction actions",
).assign_functions(fn_list)

# see the auto-generated names of your functions :)
agent.list_functions()

# Run the agent to click a specific location on the screen
output = agent.run("Click the screen at coordinates 100 and 200")
print(f"Agent: {output}")

# Run the agent to take a screenshot
output = agent.run("Take a screenshot at 1")
print(f"Agent: {output}")