import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load .env file values
load_dotenv()

# Retrieve DEBUG value from .env or default to False
DEBUG = os.getenv("DEBUG", "False").lower() in ["true", "1", "yes"]

# Example usage of DEBUG
if DEBUG:
    print("Debugging is enabled.")
else:
    print("Debugging is disabled.")
