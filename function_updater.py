import asyncio
import time
import requests
import hashlib
import os
from typing import Optional, Dict
from dotenv import load_dotenv
import aiohttp  # For asynchronous HTTP requests
import aiofiles  # For asynchronous file operations


async def file_hash(filename: str) -> str:
    """Calculate SHA256 hash of a file asynchronously."""
    hasher = hashlib.sha256()
    try:
        async with aiofiles.open(filename, "rb") as file:
            while True:
                chunk = await file.read(4096)
                if not chunk:
                    break
                hasher.update(chunk)
    except FileNotFoundError:
        print(f"{timestamp()} - File not found: {filename}")
        return ""  # Return empty string if file not found
    except Exception as e:
        print(f"{timestamp()} - Error reading file {filename}: {e}")
        return ""
    return hasher.hexdigest()


async def extract_metadata_from_file(filepath: str) -> Dict[str, str]:
    """Extract metadata from the first docstring in a Python file asynchronously."""
    metadata = {}
    try:
        async with aiofiles.open(filepath, "r") as file:
            content = await file.read()

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

    except FileNotFoundError:
        print(f"{timestamp()} - File not found: {filepath}")
    except Exception as e:
        print(f"{timestamp()} - Warning: Metadata extraction error: {e}")

    return metadata


async def check_api_availability(
    api_endpoint: str, session: aiohttp.ClientSession
) -> bool:
    """Check if the API endpoint is available, asynchronously."""
    try:
        async with session.get(f"{api_endpoint}/health") as response:
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            return True
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False


async def function_exists(
    api_endpoint: str, function_id: str, api_key: str, session: aiohttp.ClientSession
) -> bool:
    """Check if a function with the given ID exists, asynchronously."""
    try:
        async with session.get(
            f"{api_endpoint}/functions/", headers={"Authorization": f"Bearer {api_key}"}
        ) as response:
            response.raise_for_status()
            functions = await response.json()
            for function in functions:
                if function["id"] == function_id:
                    return True
            return False
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        print(f"{timestamp()} - Error checking function existence: {e}")
        return False  # Assume it doesn't exist in case of error
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False


async def create_function(
    filepath: str,
    api_endpoint: str,
    function_id: str,
    function_name: str,
    function_description: str,
    api_key: str,
    session: aiohttp.ClientSession,
) -> None:
    """Create a new remote function via API, asynchronously."""
    try:
        async with aiofiles.open(filepath, "r") as file:
            code_content = await file.read()

        payload = {
            "id": function_id,
            "name": function_name,
            "content": code_content,
            "meta": {"description": function_description, "manifest": {}},
        }
        async with session.post(
            f"{api_endpoint}/functions/create",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        ) as response:
            response.raise_for_status()
            print(f"{timestamp()} - Created new function {function_id} from {filepath}")

    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        print(
            f"{timestamp()} - Creation failed for {filepath}: {type(e).__name__}: {e}. Check network connection or API status."
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


async def update_function(
    filepath: str,
    api_endpoint: str,
    function_id: str,
    function_name: str,
    function_description: str,
    api_key: str,
    session: aiohttp.ClientSession,
) -> None:
    """Update remote function via API, or create it if it doesn't exist, asynchronously."""
    try:
        async with aiofiles.open(filepath, "r") as file:
            code_content = await file.read()

        payload = {
            "id": function_id,
            "name": function_name,
            "content": code_content,
            "meta": {"description": function_description, "manifest": {}},
        }

        async with session.post(
            f"{api_endpoint}/functions/id/{function_id}/update",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        ) as response:
            response.raise_for_status()
            print(f"{timestamp()} - Updated {function_id} from {filepath}")

    except aiohttp.ClientResponseError as e:
        if e.status == 400:  # Check for 400 Bad Request
            if await function_exists(api_endpoint, function_id, api_key, session):
                print(
                    f"{timestamp()} - Update failed for {filepath}: Function exists, but update failed. Check the function in the front-end and re-enable it if necessary."
                )
            else:
                print(
                    f"{timestamp()} - Function {function_id} not found (400 error). Creating a new function."
                )
                await create_function(  # Create the function
                    filepath,
                    api_endpoint,
                    function_id,
                    function_name,
                    function_description,
                    api_key,
                    session,
                )
        else:
            print(
                f"{timestamp()} - Update failed for {filepath}: {type(e).__name__}: {e}. Check the function in the front-end and re-enable it if necessary."
            )
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        print(
            f"{timestamp()} - Update failed for {filepath}: {type(e).__name__}: {e}. Check network connection or API status."
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def timestamp() -> str:
    """Get current timestamp in standardized format."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def validate_env_vars() -> Dict[str, str]:
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


async def process_file(
    path: str,
    api_endpoint: str,
    api_key: str,
    file_states: Dict[str, Optional[str]],
    session: aiohttp.ClientSession,
):
    """Process a single file, checking for changes and updating/creating as needed."""
    if not os.path.exists(path):
        print(f"{timestamp()} - Missing file: {path}")
        return

    current_hash = await file_hash(path)
    if not current_hash:  # if file_hash returns empty string, skip
        return
    metadata = await extract_metadata_from_file(path)

    function_id = metadata.get("id", "")
    function_name = metadata.get("title", "")
    description = metadata.get("description", "")

    if not all([function_id, function_name, description]):
        print(f"{timestamp()} - Missing metadata for {path}")
        return

    if current_hash != file_states[path]:
        if file_states[path] is not None:  # Skip initial state
            await update_function(
                path,
                api_endpoint,
                function_id,
                function_name,
                description,
                api_key,
                session,
            )
        file_states[path] = current_hash


async def main() -> None:
    """Main monitoring loop."""
    load_dotenv()
    env_vars = validate_env_vars()

    filepaths = env_vars["FILEPATHS"]
    api_endpoint = env_vars["API_ENDPOINT"]
    api_key = env_vars["API_KEY"]
    polling_interval = int(env_vars["POLLING_INTERVAL"])

    file_states: Dict[str, Optional[str]] = {path: None for path in filepaths}

    async with aiohttp.ClientSession() as session:
        while True:
            # Wait for API availability
            while not await check_api_availability(api_endpoint, session):
                print(f"{timestamp()} - API unavailable. Waiting...")
                await asyncio.sleep(polling_interval)  # Use asyncio.sleep

            # Create a list of tasks for processing each file
            tasks = [
                process_file(
                    path, api_endpoint + "/api/v1", api_key, file_states, session
                )
                for path in filepaths
            ]
            await asyncio.gather(*tasks)  # Run file processing concurrently

            await asyncio.sleep(polling_interval)


if __name__ == "__main__":
    asyncio.run(main())
