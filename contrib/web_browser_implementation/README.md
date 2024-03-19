# Web Browser Implementation
This example uses the Taskgen framework to create a web browser agent that can interact with web pages.     
The agent can take screenshots, extract text, and interact with interactive elements on the page.

## Introduction

## Features
The agent will be able to perform the following tasks:
- Open a web page
- Going to a specific URL
- Take a screenshot of the current page
- Clicking on the interactive elements

For more details, refer to the tasks list:

## Prerequisites
- Python 3.10
- taskgen library
- python-dotenv library
- OpenAI API key

## Setup
Install the required dependencies:

```bash
pip install -r requirements.txt
```
Create a `.env` file in the project root directory and add your OpenAI API key:

```bash
OPENAI_API_KEY=your_api_key_here
```

## Usage
1. Run `python web_agent.py`