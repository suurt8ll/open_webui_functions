import json
import requests
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

url = "https://api.venice.ai/api/v1/models"

# Get the API token from environment variables
api_token = os.getenv("VENICE_API_TOKEN")

headers = {"Authorization": f"Bearer {api_token}"}

response = requests.request("GET", url, headers=headers)

if response.status_code == 200:
    print(json.dumps(response.json(), indent=4))
else:
    print(f"Failed to retrieve data: {response.status_code}")
