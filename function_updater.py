import time
import requests
import hashlib
import os
from typing import Optional
from dotenv import load_dotenv


def file_hash(filename: str) -> str:
    """Calculate SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filename, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_metadata_from_file(filepath: str) -> dict[str, str]:
    """Extract metadata from the first docstring in a Python file."""
    metadata = {}
    try:
        with open(filepath, "r") as file:
            content = file.read()

        # Find first docstring using basic pattern matching
        doc_start = content.find('"""')
        if doc_start == -1:
            return metadata

        doc_end = content.find('"""', doc_start + 3)
        if doc_end == -1:
            return metadata

        docstring = content[doc_start + 3 : doc_end]
        for line in docstring.splitlines():
            line = line.strip()
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip()

    except Exception as e:
        print(f"{timestamp()} - Warning: Metadata extraction error: {e}")

    return metadata


def check_api_availability(api_endpoint: str) -> bool:
    """Check if the API endpoint is available."""
    try:
        response = requests.get(f"{api_endpoint}/health")  # Use a simple endpoint
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return True
    except requests.exceptions.RequestException:
        return False


def function_exists(api_endpoint: str, function_id: str, api_key: str) -> bool:
    """Check if a function with the given ID exists."""
    try:
        response = requests.get(
            f"{api_endpoint}/functions/", headers={"Authorization": f"Bearer {api_key}"}
        )
        response.raise_for_status()
        functions = response.json()
        for function in functions:
            if function["id"] == function_id:
                return True
        return False
    except requests.exceptions.RequestException as e:
        print(f"{timestamp()} - Error checking function existence: {e}")
        return False  # Assume it doesn't exist in case of error


def create_function(
    filepath: str,
    api_endpoint: str,
    function_id: str,
    function_name: str,
    function_description: str,
    api_key: str,
) -> None:
    """Create a new remote function via API."""
    try:
        with open(filepath, "r") as file:
            code_content = file.read()

        payload = {
            "id": function_id,
            "name": function_name,
            "content": code_content,
            "meta": {"description": function_description, "manifest": {}},
        }
        response = requests.post(
            f"{api_endpoint}/functions/create",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        )
        response.raise_for_status()
        print(f"{timestamp()} - Created new function {function_id} from {filepath}")

    except requests.exceptions.RequestException as e:
        print(
            f"{timestamp()} - Creation failed for {filepath}: {type(e).__name__}: {e}. Check network connection or API status."
        )


def update_function(
    filepath: str,
    api_endpoint: str,
    function_id: str,
    function_name: str,
    function_description: str,
    api_key: str,
) -> None:
    """Update remote function via API, or create it if it doesn't exist."""
    try:
        with open(filepath, "r") as file:
            code_content = file.read()

        payload = {
            "id": function_id,
            "name": function_name,
            "content": code_content,
            "meta": {"description": function_description, "manifest": {}},
        }

        response = requests.post(
            f"{api_endpoint}/functions/id/{function_id}/update",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        )
        response.raise_for_status()
        print(f"{timestamp()} - Updated {function_id} from {filepath}")

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:  # Check for 400 Bad Request
            if function_exists(api_endpoint + "/api/v1", function_id, api_key):
                print(
                    f"{timestamp()} - Update failed for {filepath}: Function exists, but update failed. Check the function in the front-end and re-enable it if necessary."
                )
            else:
                print(
                    f"{timestamp()} - Function {function_id} not found (400 error). Creating a new function."
                )
                create_function(  # Create the function
                    filepath,
                    api_endpoint,
                    function_id,
                    function_name,
                    function_description,
                    api_key,
                )
        else:
            print(
                f"{timestamp()} - Update failed for {filepath}: {type(e).__name__}: {e}. Check the function in the front-end and re-enable it if necessary."
            )
    except requests.exceptions.RequestException as e:
        print(
            f"{timestamp()} - Update failed for {filepath}: {type(e).__name__}: {e}. Check network connection or API status."
        )


def timestamp() -> str:
    """Get current timestamp in standardized format."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def validate_env_vars() -> dict[str, str]:
    """Validate required environment variables."""
    required_vars = ["API_ENDPOINT", "API_KEY", "FILEPATHS"]
    env_vars = {}

    for var in required_vars:
        value = os.getenv(var)
        if not value:
            print(f"{timestamp()} - Missing required environment variable: {var}")
            exit(1)
        env_vars[var] = value

    # Handle FILEPATHS as a comma-separated string
    env_vars["FILEPATHS"] = [path.strip() for path in env_vars["FILEPATHS"].split(",")]

    # Polling interval with default
    env_vars["POLLING_INTERVAL"] = int(os.getenv("POLLING_INTERVAL", "5"))

    return env_vars


def main() -> None:
    """Main monitoring loop."""
    load_dotenv()
    env_vars = validate_env_vars()

    filepaths = env_vars["FILEPATHS"]
    api_endpoint = env_vars["API_ENDPOINT"]
    api_key = env_vars["API_KEY"]
    polling_interval = env_vars["POLLING_INTERVAL"]

    file_states: dict[str, Optional[str]] = {path: None for path in filepaths}

    while True:
        # Wait for API availability
        while not check_api_availability(api_endpoint):
            print(f"{timestamp()} - API unavailable. Waiting...")
            time.sleep(int(polling_interval))
            #  keep trying every polling_interval

        for path in filepaths:
            if not os.path.exists(path):
                print(f"{timestamp()} - Missing file: {path}")
                continue

            current_hash = file_hash(path)
            metadata = extract_metadata_from_file(path)

            # FIXME handle missing metadata better
            function_id = metadata.get("id", "")
            function_name = metadata.get("title", "")
            description = metadata.get("description", "")

            if not all([function_id, function_name, description]):
                print(f"{timestamp()} - Missing metadata for {path}")
                continue

            if current_hash != file_states[path]:
                if file_states[path] is not None:  # Skip initial state
                    update_function(
                        path,
                        api_endpoint + "/api/v1",
                        function_id,
                        function_name,
                        description,
                        api_key,
                    )
                file_states[path] = current_hash

        time.sleep(int(polling_interval))


if __name__ == "__main__":
    main()
