import requests
from dotenv import load_dotenv
import os
import base64
from io import BytesIO
from PIL import Image

load_dotenv()

api_token = os.getenv("VENICE_API_TOKEN")
headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}


def get_models():
    """Retrieves available models from the Venice.ai API."""
    url = "https://api.venice.ai/api/v1/models"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()["data"]
    else:
        print(f"Failed to retrieve models: {response.status_code}")
        return []


def choose_model(models):
    """Allows the user to select a model from the available list."""
    print("Available models:")
    image_models = [model for model in models if model["type"] == "image"]
    for i, model in enumerate(image_models):
        print(f"{i + 1}. {model['id']}")
    while True:
        try:
            choice = int(input("Select a model by its number: "))
            if 1 <= choice <= len(image_models):
                return image_models[choice - 1]["id"]
            else:
                print("Invalid choice. Please enter a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def generate_image(prompt, model, width=1024, height=1024, negative_prompt=""):
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


def display_and_save_image(image_data, filename="generated_image.png"):
    """Displays and saves the generated image."""
    if image_data and "images" in image_data:
        for i, image_str in enumerate(image_data["images"]):
            image_bytes = base64.b64decode(image_str)
            image = Image.open(BytesIO(image_bytes))
            image.show()
            image.save(f"{filename.split('.')[0]}_{i}.{filename.split('.')[1]}")
    else:
        print("No image data found.")


# Example usage:
if __name__ == "__main__":
    models = get_models()
    selected_model = choose_model(models)
    prompt = "a cat wearing a hat"
    image_data = generate_image(prompt, selected_model)
    display_and_save_image(image_data)
