import os
import sys
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

# Retrieve API key
api_key = os.getenv("GEMINI_API_KEY", "")

# Handle missing API key
if not api_key:
    print("Error: Missing API key. Please set 'GEMINI_API_KEY' in your environment variables.")
    sys.exit(1)

# Retrieve model name from .env
model_name = os.getenv("MODEL_NAME", "gemini-2.0-flash-thinking-exp").strip()

# Handle missing model name
if not model_name:
    print("Error: Missing model name. Please set 'MODEL_NAME' in your environment variables.")
    sys.exit(1)

if DEBUG:
    print("Using model:", model_name)

client = genai.Client(api_key=api_key, http_options={'api_version':'v1alpha'})

# Set a default system instruction
DEFAULT_SYSTEM_PROMPT = "You are a helpful and intelligent assistant. Provide accurate and concise answers."

# Prompt the user for a query
user_query = input("Enter your query: ")

config = types.GenerateContentConfig(
  system_instruction=DEFAULT_SYSTEM_PROMPT,
  temperature=0.4,
  thinking_config={'include_thoughts': True}
)

# Stream the response
is_thought = False
is_response = False

for chunk in client.models.generate_content_stream(
    model=model_name,
    contents=user_query,
    config=config
):
    for part in chunk.candidates[0].content.parts:
        if part.thought and not is_thought:
            print("\n--- Start of Model Thought ---\n")
            is_thought = True
            is_response = False
        elif not part.thought and not is_response:
            print("\n--- Start of Model Response ---\n")
            is_thought = False
            is_response = True
        print(part.text, end="")

# Ensure proper formatting at the end
if is_thought:
    print("\n--- End of Model Thought ---\n")
elif is_response:
    print("\n--- End of Model Response ---\n")
