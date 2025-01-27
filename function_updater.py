import time
import json
import requests
import hashlib
import argparse


def file_hash(filename):
    """Calculate SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filename, "rb") as file:
        while True:
            chunk = file.read(4096)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def update_function(
    filepath,
    api_endpoint,
    function_id,
    function_name,
    function_description,
    manifest,
    api_key,
):
    """Read file, format JSON, and POST to API with API key."""
    try:
        with open(filepath, "r") as file:
            code_content = file.read()

        request_body = {
            "id": function_id,
            "name": function_name,
            "content": code_content,
            "meta": {"description": function_description, "manifest": manifest},
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",  # Add API key to headers - adjust header name if needed
        }
        update_url = f"{api_endpoint}/functions/id/{function_id}/update"

        response = requests.post(
            update_url, data=json.dumps(request_body), headers=headers
        )
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        print(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Function '{function_id}' updated successfully."
        )

    except FileNotFoundError:
        print(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Error: File not found at '{filepath}'"
        )
    except requests.exceptions.RequestException as e:
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - API request failed: {e}")
    except Exception as e:
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Monitors a Python file and updates a function via API on changes."
    )
    parser.add_argument("filepath", help="Path to the Python code file")
    parser.add_argument(
        "api_endpoint",
        help="Base API endpoint URL (e.g., http://localhost:8080/api/v1)",
    )
    parser.add_argument("function_id", help="ID of the function to update")
    parser.add_argument("api_key", help="Your API Key")  # Added API Key argument
    parser.add_argument(
        "--name",
        default="Gemini Manifold (google-genai)",
        help="Function name (default: Gemini Manifold (google-genai))",
    )
    parser.add_argument(
        "--description",
        default="Manifold function for Gemini Developer API. Uses google-genai, supports thinking models.",
        help="Function description",
    )
    parser.add_argument(
        "--manifest_title",
        default="Gemini Manifold (google-genai)",
        help="Manifest title",
    )
    parser.add_argument("--manifest_author", default="suurt8ll", help="Manifest author")
    parser.add_argument(
        "--manifest_author_url",
        default="https://github.com/suurt8ll",
        help="Manifest author_url",
    )
    parser.add_argument(
        "--manifest_funding_url",
        default="https://github.com/suurt8ll/open_webui_functions",
        help="Manifest funding_url",
    )
    parser.add_argument("--manifest_version", default="0.1.0", help="Manifest version")
    parser.add_argument(
        "--polling_interval",
        type=int,
        default=5,
        help="File change polling interval in seconds (default: 5)",
    )

    args = parser.parse_args()

    previous_hash = None

    while True:
        current_hash = file_hash(args.filepath)
        if current_hash != previous_hash:
            if (
                previous_hash is not None
            ):  # Avoid updating on initial run if file exists
                manifest_data = {
                    "title": args.manifest_title,
                    "author": args.manifest_author,
                    "author_url": args.manifest_author_url,
                    "funding_url": args.manifest_funding_url,
                    "version": args.manifest_version,
                }
                update_function(
                    args.filepath,
                    args.api_endpoint,
                    args.function_id,
                    args.name,
                    args.description,
                    manifest_data,
                    args.api_key,  # Pass API Key to update_function
                )
            previous_hash = current_hash
        time.sleep(args.polling_interval)
