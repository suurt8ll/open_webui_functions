import asyncio
import hashlib
import os
import signal
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import aiofiles
import aiohttp
from dotenv import load_dotenv
from loguru import logger

# --- Configuration & State Management ---


class ApiState(Enum):
    """Represents the perceived state of the remote API."""

    UNKNOWN = auto()
    AVAILABLE = auto()
    UNAVAILABLE = auto()


@dataclass(frozen=True)
class Config:
    """Typed configuration loaded from environment variables."""

    api_endpoint: str
    api_key: str
    filepaths: list[str]
    polling_interval: int


def load_config() -> Config:
    """Loads and validates required environment variables into a Config object."""
    load_dotenv()
    required_vars = ["API_ENDPOINT", "API_KEY", "FILEPATHS"]
    env_vars: dict[str, Any] = {}

    for var in required_vars:
        value = os.getenv(var)
        if not value:
            logger.critical(f"Missing required environment variable: {var}")
            raise ValueError(f"Missing environment variable: {var}")

        env_vars[var] = value

    return Config(
        api_endpoint=env_vars["API_ENDPOINT"],
        api_key=env_vars["API_KEY"],
        filepaths=[path.strip() for path in env_vars["FILEPATHS"].split(",")],
        polling_interval=int(os.getenv("POLLING_INTERVAL", "5")),
    )


# --- File Utilities ---


async def file_hash(filename: str) -> str | None:
    """
    Calculate the SHA256 hash of a file asynchronously.
    Returns None if the file cannot be read.
    """
    hasher = hashlib.sha256()
    try:
        async with aiofiles.open(filename, "rb") as file:
            while chunk := await file.read(4096):
                hasher.update(chunk)
    except FileNotFoundError:
        # This is handled separately in the main loop, so no need to log here.
        return None
    except Exception as e:
        logger.exception(f"Error reading file {filename}: {e}")
        return None
    return hasher.hexdigest()


async def extract_metadata_from_file(filepath: str) -> dict[str, str]:
    """
    Extracts metadata from the first docstring in a Python file.

    NOTE: This is a simple parser. For more complex scenarios, using `ast.get_docstring`
          might be more robust, but this is sufficient for this utility's purpose.
    """
    metadata = {}
    try:
        async with aiofiles.open(filepath, "r", encoding="utf-8") as file:
            content = await file.read()

        # A simple way to find the first module-level docstring
        if not content.strip().startswith('"""'):
            return {}

        doc_start = content.find('"""')
        doc_end = content.find('"""', doc_start + 3)

        if doc_end == -1:
            return {}

        docstring = content[doc_start + 3 : doc_end]
        for line in docstring.splitlines():
            line = line.strip()
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip()

    except FileNotFoundError:
        # This is handled by the caller, no need to log here.
        pass
    except Exception as e:
        logger.warning(f"Metadata extraction error for {filepath}: {e}")

    return metadata


# --- API Client ---


class OwuiApiClient:
    """A client to interact with the OWUI functions API."""

    def __init__(self, session: aiohttp.ClientSession, root_url: str, api_key: str):
        self._session = session
        self._root_url = root_url
        self._base_api_url = f"{root_url}/api/v1"
        self._api_key = api_key
        self._headers = {"Authorization": f"Bearer {self._api_key}"}

    async def check_availability(self) -> bool:
        """Checks if the API is healthy and responsive."""
        try:
            # The health endpoint is at the root, not under /api/v1
            async with self._session.get(
                f"{self._root_url}/health", timeout=aiohttp.ClientTimeout(total=3)
            ) as response:
                return response.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False
        except Exception as e:
            logger.exception(f"Unexpected error checking API: {e}")
            return False

    async def function_exists(self, function_id: str) -> bool:
        """Checks if a function with the given ID exists."""
        try:
            async with self._session.get(
                f"{self._base_api_url}/functions/", headers=self._headers
            ) as response:
                response.raise_for_status()
                functions = await response.json()
                return any(f["id"] == function_id for f in functions)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.error(f"Error checking function existence: {e}")
            return False
        except Exception as e:
            logger.exception(f"Unexpected error checking function: {e}")
            return False

    async def _create_function(
        self,
        filepath: str,
        function_id: str,
        function_name: str,
        description: str,
        code_content: str,
    ) -> None:
        """Helper to create a new remote function."""
        payload = {
            "id": function_id,
            "name": function_name,
            "content": code_content,
            "meta": {"description": description, "manifest": {}},
        }
        async with self._session.post(
            f"{self._base_api_url}/functions/create",
            headers=self._headers,
            json=payload,
        ) as response:
            response.raise_for_status()
            logger.info(f"✅ Created new function '{function_id}' from {filepath}")

    async def update_or_create_function(
        self,
        filepath: str,
        function_id: str,
        function_name: str,
        description: str,
    ) -> None:
        """Updates a remote function, or creates it if it doesn't exist."""
        try:
            async with aiofiles.open(filepath, "r", encoding="utf-8") as file:
                code_content = await file.read()
        except (IOError, OSError) as e:
            logger.error(f"Failed to read file {filepath} before update: {e}")
            return

        try:
            payload = {
                "id": function_id,
                "name": function_name,
                "content": code_content,
                "meta": {"description": description, "manifest": {}},
            }

            async with self._session.post(
                f"{self._base_api_url}/functions/id/{function_id}/update",
                headers=self._headers,
                json=payload,
            ) as response:
                response.raise_for_status()
                logger.info(f"🔄 Updated function '{function_id}' from {filepath}")

        except aiohttp.ClientResponseError as e:
            if e.status in {400, 404}:  # Handle both 'Bad Request' and 'Not Found'
                logger.warning(
                    f"Function '{function_id}' not found or failed to update (Status: {e.status}). Attempting to create it."
                )
                try:
                    await self._create_function(
                        filepath,
                        function_id,
                        function_name,
                        description,
                        code_content,
                    )
                except (aiohttp.ClientError, asyncio.TimeoutError) as create_err:
                    logger.error(f"Creation failed for {filepath}: {create_err}")
                except Exception as create_exc:
                    logger.exception(
                        f"Unexpected error creating function {function_id}"
                    )
            else:
                logger.error(f"Update failed for {filepath}: {e}")
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.error(
                f"Update failed for {filepath}: {type(e).__name__}: {e}. Check network/API status."
            )
        except Exception as e:
            logger.exception(f"Unexpected error updating function: {e}")


# --- Main Application ---


class FunctionUpdater:
    """The main application class that orchestrates file monitoring and updates."""

    def __init__(self, config: Config, shutdown_event: asyncio.Event):
        self.config = config
        self.shutdown_event = shutdown_event
        self.api_state = ApiState.UNKNOWN
        self.file_hashes: dict[str, str | None] = {
            path: None for path in config.filepaths
        }
        self.missing_file_logged: set[str] = set()

    async def _check_api_status(self, client: OwuiApiClient) -> None:
        """Checks API availability and updates the state, logging changes."""
        is_available = await client.check_availability()
        if is_available:
            if self.api_state != ApiState.AVAILABLE:
                logger.info("API is available. Starting to process files...")
                self.api_state = ApiState.AVAILABLE
        else:
            if self.api_state != ApiState.UNAVAILABLE:
                logger.warning("API is unavailable. Will keep retrying...")
                self.api_state = ApiState.UNAVAILABLE

    async def _process_file(self, path: str, client: OwuiApiClient) -> None:
        """Processes a single file, checking for changes and updating if necessary."""
        if not os.path.exists(path):
            if path not in self.missing_file_logged:
                logger.warning(f"File not found: {path}")
                self.missing_file_logged.add(path)
            return

        if path in self.missing_file_logged:
            self.missing_file_logged.remove(path)

        current_hash = await file_hash(path)
        if not current_hash:
            return

        if current_hash == self.file_hashes.get(path):
            return  # No change, skip processing

        logger.debug(f"File change detected for {path}")
        metadata = await extract_metadata_from_file(path)
        function_id = metadata.get("id")
        function_name = metadata.get("title")
        description = metadata.get("description")

        if not all([function_id, function_name, description]):
            logger.warning(
                f"Skipping {path}: Missing required metadata ('id', 'title', 'description')."
            )
            return

        # Assertions to satisfy the type checker after the 'all' check.
        assert function_id is not None
        assert function_name is not None
        assert description is not None

        # A file has changed, so we trigger an update.
        await client.update_or_create_function(
            path, function_id, function_name, description
        )
        # Only update the hash on successful processing to ensure retry on failure.
        self.file_hashes[path] = current_hash

    async def run(self) -> None:
        """The main execution loop."""
        async with aiohttp.ClientSession() as session:
            api_client = OwuiApiClient(
                session, self.config.api_endpoint, self.config.api_key
            )
            while not self.shutdown_event.is_set():
                await self._check_api_status(api_client)

                if self.api_state == ApiState.AVAILABLE:
                    # Use a TaskGroup for robust concurrent execution (Python 3.11+)
                    try:
                        async with asyncio.TaskGroup() as tg:
                            for path in self.config.filepaths:
                                tg.create_task(self._process_file(path, api_client))
                    except Exception as e:
                        logger.exception(
                            f"An error occurred during file processing: {e}"
                        )

                try:
                    # Use the shutdown event for a more responsive sleep
                    await asyncio.wait_for(
                        self.shutdown_event.wait(), timeout=self.config.polling_interval
                    )
                except asyncio.TimeoutError:
                    continue  # This is the normal polling loop
                except asyncio.CancelledError:
                    break


async def main_wrapper() -> None:
    """Sets up signal handling and runs the main application."""
    shutdown_event = asyncio.Event()

    def signal_handler(signum, frame):
        logger.warning(f"Received signal {signum}. Initiating shutdown...")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler, sig, None)

    try:
        config = load_config()
        updater = FunctionUpdater(config, shutdown_event)
        await updater.run()
    except ValueError as e:
        # This catches configuration errors from load_config
        logger.critical(e)
    except asyncio.CancelledError:
        logger.info("Main task cancelled.")
    except Exception:
        logger.exception("An unexpected error occurred in the main application:")
    finally:
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    logger.info("Starting function updater utility...")
    asyncio.run(main_wrapper())
