import requests
from dotenv import load_dotenv
import os
import base64
from io import BytesIO
from PIL import Image

load_dotenv()

api_token = os.getenv("VENICE_API_TOKEN")
headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}


def generate_image(
    prompt, model="fluently-xl", width=512, height=512, negative_prompt=""
):
    """Generates an image using the Venice.ai API."""
    url = "https://api.venice.ai/api/v1/image/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "width": width,
        "height": height,
        "steps": 30,
        "hide_watermark": True,
        "return_binary": False,
        "seed": 123,
        "cfg_scale": 7,
        # "style_preset": "3D Model",
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
            image.show()  # Display the image
            image.save(
                f"{filename.split('.')[0]}_{i}.{filename.split('.')[1]}"
            )  # Save the image
    else:
        print("No image data found.")


# Example usage:
if __name__ == "__main__":
    prompt = "a fish wearing a hat"
    image_data = generate_image(prompt)
    display_and_save_image(image_data)
