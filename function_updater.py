import hashlib
import os
from typing import Optional
from dotenv import load_dotenv
import aiohttp  # For asynchronous HTTP requests
import aiofiles  # For asynchronous file operations
import logging
import asyncio
import coloredlogs
import signal  # Import the signal module for handling signals


class CustomFormatter(logging.Formatter):
    """Custom formatter to add thread name."""

    def format(self, record):
        if asyncio.iscoroutinefunction(record.funcName):
            record.funcName = f"{record.funcName} [coroutine]"
        return super().format(record)


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set the root logger level

# Create a handler for console output
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Set the console handler level

# Create a formatter and set it for the handler
formatter = CustomFormatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s"
)
console_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(console_handler)

# Install coloredlogs with custom format and level styles
coloredlogs.install(
    level="DEBUG",
    logger=logger,
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s",
    level_styles={
        "debug": {"color": "white"},
        "info": {"color": "green"},
        "warning": {"color": "yellow"},
        "error": {"color": "red"},
        "critical": {"color": "red", "bold": True},
    },
    field_styles={
        "asctime": {"color": "blue"},
        "name": {"color": "cyan"},
        "levelname": {"color": "white"},
        "funcName": {"color": "cyan"},
    },
)


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
        logger.error(f"File not found: {filename}")
        return ""  # Return empty string if file not found
    except Exception as e:
        logger.exception(
            f"Error reading file {filename}: {e}"
        )  # Use exception for full traceback
        return ""
    return hasher.hexdigest()


async def extract_metadata_from_file(filepath: str) -> dict[str, str]:
    """Extract metadata from the first docstring in a Python file asynchronously."""
    metadata = {}
    try:
        async with aiofiles.open(filepath, "r") as file:
            content = await file.read()

        # Find first docstring
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
        logger.error(f"File not found: {filepath}")
    except Exception as e:
        logger.warning(f"Metadata extraction error: {e}")

    return metadata


async def check_api_availability(
    api_endpoint: str, session: aiohttp.ClientSession
) -> bool:
    """Check if the API endpoint is available, asynchronously."""
    try:
        async with session.get(f"{api_endpoint}/health") as response:
            response.raise_for_status()
            return True
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        return False
    except Exception as e:
        logger.exception(f"Unexpected error checking API: {e}")
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
        logger.error(f"Error checking function existence: {e}")
        return False
    except Exception as e:
        logger.exception(f"Unexpected error checking function: {e}")
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
            logger.info(f"Created new function {function_id} from {filepath}")

    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.error(
            f"Creation failed for {filepath}: {type(e).__name__}: {e}. Check network/API status."
        )
    except Exception as e:
        logger.exception(f"Unexpected error creating function: {e}")


async def update_function(
    filepath: str,
    api_endpoint: str,
    function_id: str,
    function_name: str,
    function_description: str,
    api_key: str,
    session: aiohttp.ClientSession,
) -> None:
    """Update remote function, or create if it doesn't exist, asynchronously."""
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
            logger.info(f"Updated {function_id} from {filepath}")

    except aiohttp.ClientResponseError as e:
        if e.status == 400:
            if await function_exists(api_endpoint, function_id, api_key, session):
                logger.error(
                    f"Update failed for {filepath}: Function exists, but update failed. Check function and re-enable."
                )
            else:
                logger.warning(
                    f"Function {function_id} not found (400). Creating new function."
                )
                await create_function(
                    filepath,
                    api_endpoint,
                    function_id,
                    function_name,
                    function_description,
                    api_key,
                    session,
                )
        else:
            logger.error(
                f"Update failed for {filepath}: {type(e).__name__}: {e}. Check function and re-enable."
            )
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.error(
            f"Update failed for {filepath}: {type(e).__name__}: {e}. Check network/API status."
        )
    except Exception as e:
        logger.exception(f"Unexpected error updating function: {e}")


def validate_env_vars() -> dict[str, str]:
    """Validate required environment variables."""
    required_vars = ["API_ENDPOINT", "API_KEY", "FILEPATHS"]
    env_vars = {}

    for var in required_vars:
        value = os.getenv(var)
        if not value:
            logger.critical(f"Missing required environment variable: {var}")
            exit(1)  # Exit with a non-zero code to indicate failure
        env_vars[var] = value

    env_vars["FILEPATHS"] = [path.strip() for path in env_vars["FILEPATHS"].split(",")]
    env_vars["POLLING_INTERVAL"] = int(os.getenv("POLLING_INTERVAL", "5"))

    return env_vars


async def process_file(
    path: str,
    api_endpoint: str,
    api_key: str,
    file_states: dict[str, Optional[str]],
    session: aiohttp.ClientSession,
):
    """Process a single file, checking for changes and updating/creating."""
    if not os.path.exists(path):
        logger.warning(f"Missing file: {path}")
        return

    current_hash = await file_hash(path)
    if not current_hash:
        return
    metadata = await extract_metadata_from_file(path)

    function_id = metadata.get("id", "")
    function_name = metadata.get("title", "")
    description = metadata.get("description", "")

    if not all([function_id, function_name, description]):
        logger.warning(f"Missing metadata for {path}")
        return

    if current_hash != file_states[path]:
        if file_states[path] is not None:
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

    file_states: dict[str, Optional[str]] = {path: None for path in filepaths}

    api_available = True  # Assume API is available at startup
    api_unavailable_logged = False  # Flag to track if "API unavailable" has been logged

    async with aiohttp.ClientSession() as session:
        while True:
            if shutdown_event.is_set():  # Check for shutdown signal
                logger.info("Exiting main loop due to shutdown signal.")
                break

            if not await check_api_availability(api_endpoint, session):
                if api_available:  # Only log if API was previously available
                    logger.info(f"API unavailable. Waiting...")
                    api_available = False
                    api_unavailable_logged = (
                        True  # Mark that unavailable log has been made
                    )
            else:
                if (
                    not api_available and api_unavailable_logged
                ):  # Log reconnection only if it was previously unavailable and log was made
                    logger.info("API reconnected.")
                    api_available = True
                    api_unavailable_logged = False  # Reset the unavailable log flag

            if api_available:  # Only process files if API is available
                tasks = [
                    process_file(
                        path, api_endpoint + "/api/v1", api_key, file_states, session
                    )
                    for path in filepaths
                ]
                await asyncio.gather(*tasks)

            await asyncio.sleep(polling_interval)


if __name__ == "__main__":
    shutdown_event = asyncio.Event()  # Create an event for shutdown signal

    def signal_handler(signum, frame):
        logger.warning(f"Received signal {signum}. Shutting down...")
        shutdown_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler, sig, None)

    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
