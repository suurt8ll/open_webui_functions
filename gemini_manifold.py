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

# Retrieve API key and handle missing key
api_key = os.getenv("GEMINI_API_KEY", "")
if not api_key:
    print("Error: Missing API key. \
            Please set the GEMINI_API_KEY in your environment variables.")
    exit(1)

client = genai.Client(api_key=api_key)
