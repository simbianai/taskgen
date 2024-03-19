import os
from dotenv import load_dotenv

def load_api_key(dotenv_path='../../.env'):
    load_dotenv(dotenv_path)
    return os.getenv('OPENAI_API_KEY')

