import os
import sys
import datetime

# Constants
# Assuming this script is in utils/, and we want to go up one level then into plugins/pipes
TARGET_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'plugins', 'pipes', 'gemini_models.yaml')

def get_input(prompt, default=None, required=True):
    """Get input from user with optional default and validation."""
    while True:
        if default is not None:
            user_input = input(f"{prompt} [{default}]: ").strip()
        else:
            user_input = input(f"{prompt}: ").strip()

        if not user_input:
            if default is not None:
                return default
            if not required:
                return ""
            print("Error: This field is required.")
        else:
            return user_input

def get_list_input(prompt, default_list):
    """Get a list of items from user."""
    default_str = ", ".join(default_list)
    user_input = input(f"{prompt} [{default_str}]: ").strip()
    if not user_input:
        return default_list
    return [item.strip() for item in user_input.split(',')]

def get_bool_input(prompt, default=False):
    """Get boolean input."""
    default_str = "Y/n" if default else "y/N"
    user_input = input(f"{prompt} ({default_str}): ").strip().lower()
    if not user_input:
        return default
    return user_input.startswith('y')

def generate_yaml_entry(data):
    """Generate YAML string for the model."""
    # Helper to format lists
    def format_list(lst):
        return "[" + ", ".join(lst) + "]"

    # Helper to format pricing
    def format_pricing_tier(tier):
        up_to = tier['up_to_tokens']
        if up_to is None:
            up_to = "null"
        return f"""      - up_to_tokens: {up_to}
        price_per_million: {tier['price_per_million']}"""

    entry = f"""
{data['model_id']}:
  description: "{data['description']}"
  knowledge_cutoff: "{data['knowledge_cutoff']}"
  latest_update: "{data['latest_update']}"
  supported_data_types:
    inputs: {format_list(data['inputs'])}
    outputs: {format_list(data['outputs'])}
  tokens:
    input: {data['input_tokens']}
    output: {data['output_tokens']}
  capabilities:
    audio_generation: {str(data['caps']['audio_generation']).lower()}
    batch_api: {str(data['caps']['batch_api']).lower()}
    caching: {str(data['caps']['caching']).lower()}
    code_execution: {str(data['caps']['code_execution']).lower()}
    file_search: {str(data['caps']['file_search']).lower()}
    function_calling: {str(data['caps']['function_calling']).lower()}
    grounding_google_maps: {str(data['caps']['grounding_google_maps']).lower()}
    image_generation: {str(data['caps']['image_generation']).lower()}
    live_api: {str(data['caps']['live_api']).lower()}
    search_grounding: {str(data['caps']['search_grounding']).lower()}
    structured_outputs: {str(data['caps']['structured_outputs']).lower()}
    thinking: {str(data['caps']['thinking']).lower()}
    url_context: {str(data['caps']['url_context']).lower()}
  pricing:
    input:
{chr(10).join([format_pricing_tier(t) for t in data['pricing']['input']])}
    output:
{chr(10).join([format_pricing_tier(t) for t in data['pricing']['output']])}"""

    if data['pricing'].get('caching'):
        entry += f"""
    caching:
{chr(10).join([format_pricing_tier(t) for t in data['pricing']['caching']])}"""

    if data['pricing'].get('image_output'):
        entry += f"""
    image_output:
{chr(10).join([format_pricing_tier(t) for t in data['pricing']['image_output']])}"""

    return entry + "\n"

def main():
    print("--- Gemini Model Wizard ---")
    print(f"Target file: {TARGET_FILE}")

    if not os.path.exists(TARGET_FILE):
        print("Error: Target file not found!")
        sys.exit(1)

    # Collect Data
    model_id = get_input("Model ID (e.g., gemini-3-pro)")
    description = get_input("Description")

    today = datetime.date.today()
    knowledge_cutoff = get_input("Knowledge Cutoff (YYYY-MM)", default=f"{today.year}-01")
    latest_update = get_input("Latest Update (YYYY-MM)", default=f"{today.year}-{today.month:02d}")

    inputs = get_list_input("Supported Inputs", ["Text", "Image", "Video", "Audio", "PDF"])
    outputs = get_list_input("Supported Outputs", ["Text"])

    input_tokens = get_input("Input Token Limit", default="1048576")
    output_tokens = get_input("Output Token Limit", default="8192")

    print("\n--- Capabilities ---")
    caps = {
        "audio_generation": get_bool_input("Audio Generation", False),
        "batch_api": get_bool_input("Batch API", True),
        "caching": get_bool_input("Caching", True),
        "code_execution": get_bool_input("Code Execution", False),
        "file_search": get_bool_input("File Search", False),
        "function_calling": get_bool_input("Function Calling", True),
        "grounding_google_maps": get_bool_input("Grounding (Google Maps)", False),
        "image_generation": get_bool_input("Image Generation", False),
        "live_api": get_bool_input("Live API", False),
        "search_grounding": get_bool_input("Search Grounding", False),
        "structured_outputs": get_bool_input("Structured Outputs", True),
        "thinking": get_bool_input("Thinking", False),
        "url_context": get_bool_input("URL Context", False),
    }

    print("\n--- Pricing ---")
    # Simplified pricing input for wizard
    input_price = get_input("Input Price per Million (up to 128k/null)", default="0.10")
    output_price = get_input("Output Price per Million (up to 128k/null)", default="0.40")

    pricing = {
        "input": [{"up_to_tokens": None, "price_per_million": input_price}],
        "output": [{"up_to_tokens": None, "price_per_million": output_price}]
    }

    if caps['caching']:
        try:
            default_cache = str(float(input_price)/4)
        except ValueError:
            default_cache = "0.025"
        cache_price = get_input("Cached Input Price per Million", default=default_cache)
        pricing["caching"] = [{"up_to_tokens": None, "price_per_million": cache_price}]

    if caps['image_generation']:
        img_price = get_input("Image Output Price per Million", default="30.00")
        pricing["image_output"] = [{"up_to_tokens": None, "price_per_million": img_price}]

    data = {
        "model_id": model_id,
        "description": description,
        "knowledge_cutoff": knowledge_cutoff,
        "latest_update": latest_update,
        "inputs": inputs,
        "outputs": outputs,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "caps": caps,
        "pricing": pricing
    }

    new_entry = generate_yaml_entry(data)

    print("\n--- Generated Entry ---")
    print(new_entry)

    if not get_bool_input("Proceed with writing to file?", True):
        print("Aborted.")
        sys.exit(0)

    # Read and Insert
    with open(TARGET_FILE, 'r') as f:
        lines = f.readlines()

    # Find insertion point: After header comments
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.strip() and not line.strip().startswith('#'):
            insert_idx = i
            break
    else:
        # If no models found, append to end
        insert_idx = len(lines)

    lines.insert(insert_idx, new_entry)

    with open(TARGET_FILE, 'w') as f:
        f.writelines(lines)

    print(f"Successfully added {model_id} to {TARGET_FILE}")

if __name__ == "__main__":
    main()
