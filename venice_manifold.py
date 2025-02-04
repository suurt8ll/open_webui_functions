import json
import requests
from dotenv import load_dotenv
import os

load_dotenv()

api_token = os.getenv("VENICE_API_TOKEN")
headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}


def generate_image(
    prompt, model="fluently-xl", width=1024, height=1024, negative_prompt=""
):
    """Generates an image using the Venice.ai API."""
    url = "https://api.venice.ai/api/v1/image/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "width": width,
        "height": height,
        "steps": 30,
        "hide_watermark": False,
        "return_binary": False,
        "seed": 123,
        "cfg_scale": 7,
        "style_preset": "3D Model",
        "negative_prompt": negative_prompt,
        "safe_mode": False,
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Image generation failed: {response.status_code}")
        return None


# Example usage:
if __name__ == "__main__":
    prompt = "a cat wearing a hat"
    image_data = generate_image(prompt)
    if image_data:
        print(json.dumps(image_data, indent=4))
    else:
        print("Failed to generate image.")
