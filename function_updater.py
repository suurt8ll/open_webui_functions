import time
import json
import requests
import hashlib
import argparse
import os
from dotenv import load_dotenv


def file_hash(filename):
    hasher = hashlib.sha256()
    with open(filename, "rb") as file:
        while True:
            chunk = file.read(4096)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_metadata_from_file(filepath):
    metadata = {}
    try:
        with open(filepath, "r") as file:
            lines = file.readlines()
            in_docstring = False
            for line in lines:
                line = line.strip()
                if line == '"""':
                    if in_docstring:
                        in_docstring = False
                        break
                    else:
                        in_docstring = True
                    continue
                if in_docstring:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        metadata[key.strip()] = value.strip()
    except FileNotFoundError:
        print(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Warning: Metadata file not found at '{filepath}' while trying to extract metadata."
        )
    except Exception as e:
        print(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Warning: Error extracting metadata from file: {e}"
        )
    return metadata


def update_function(
    filepath,
    api_endpoint,
    function_id,
    function_name,
    function_description,
    api_key,
):
    try:
        with open(filepath, "r") as file:
            code_content = file.read()
        request_body = {
            "id": function_id,
            "name": function_name,
            "content": code_content,
            "meta": {"description": function_description, "manifest": {}},
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        update_url = f"{api_endpoint}/functions/id/{function_id}/update"
        response = requests.post(
            update_url, data=json.dumps(request_body), headers=headers
        )
        response.raise_for_status()
        print(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Function '{function_id}' updated successfully from file '{filepath}'."
        )
    except FileNotFoundError:
        print(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Error: File not found at '{filepath}'"
        )
    except requests.exceptions.RequestException as e:
        print(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} - API request failed for file '{filepath}': {e}"
        )
    except Exception as e:
        print(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} - An error occurred for file '{filepath}': {e}"
        )


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Monitors Python file(s) and updates function(s) via API on changes."
    )
    parser.add_argument(
        "--filepaths",
        nargs="+",
        help="Path to the Python code file(s)",
    )
    parser.add_argument(
        "--api_endpoint",
        help="Base API endpoint URL (e.g., http://localhost:8080/api/v1)",
        default=os.getenv("API_ENDPOINT"),
    )
    # FIXME I need to support defining multiple function IDs.
    parser.add_argument(
        "--function_id",
        help="ID of the function to update (default: from file metadata 'id')",
    )
    parser.add_argument("--api_key", help="Your API Key", default=os.getenv("API_KEY"))
    parser.add_argument(
        "--name",
        help="Function name (default: from file metadata 'title')",
    )
    parser.add_argument(
        "--description",
        help="Function description (default: from file metadata 'description')",
    )
    parser.add_argument(
        "--polling_interval",
        type=int,
        default=5,
        help="File change polling interval in seconds (default: 5)",
    )
    args = parser.parse_args()
    missing_args = []
    if not args.filepaths:
        missing_args.append("filepaths")
    if not args.api_endpoint:
        missing_args.append("api_endpoint")
    if not args.api_key:
        missing_args.append("api_key")
    if not args.polling_interval:
        missing_args.append("polling_interval")
    if missing_args:
        print(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Error: Missing required arguments: {', '.join(missing_args)}"
        )
        print("Please provide them via command line or .env file.")
        exit(1)
    previous_hashes = {}
    while True:
        for filepath in args.filepaths:
            metadata_from_file = extract_metadata_from_file(filepath)
            function_name = args.name or metadata_from_file.get("title")
            function_id = args.function_id or metadata_from_file.get("id")
            function_description = args.description or metadata_from_file.get(
                "description"
            )
            missing_metadata = []
            if not function_name:
                missing_metadata.append("name")
            if not function_id:
                missing_metadata.append("id")
            if not function_description:
                missing_metadata.append("description")
            if missing_metadata:
                print(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Error: Missing function metadata for '{filepath}': {', '.join(missing_metadata)}"
                )
                print(
                    "Please provide them via command line, .env file or in function file docstring."
                )
                continue
            current_hash = file_hash(filepath)
            if filepath not in previous_hashes:
                previous_hashes[filepath] = None
            if current_hash != previous_hashes[filepath]:
                if previous_hashes[filepath] is not None:
                    update_function(
                        filepath,
                        args.api_endpoint,
                        function_id,
                        function_name,
                        function_description,
                        args.api_key,
                    )
                previous_hashes[filepath] = current_hash
        time.sleep(args.polling_interval)
